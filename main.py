import os
import traceback
import logging
from dotenv import load_dotenv
from crewai import Crew, Task
from langchain_groq import ChatGroq
from agents.internet_search_agent import InternetSearchAgent
from agents.content_generator_agent import ContentGeneratorAgent
from utils.terminal_ui import TerminalUI
from config.settings import GROQ_API_KEY, SERPER_API_KEY, MONGODB_URI
from rich.console import Console
from rich.traceback import install
from tools.database_handler import DatabaseHandler
from models.task_outputs import SearchOutput, ContentOutput
import litellm
from litellm.exceptions import OpenAIError
import warnings
import re
import json

# Configure rich traceback handler
install(show_locals=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific SageMaker INFO warnings
warnings.filterwarnings("ignore", message="Not applying SDK defaults from location:.*")

# Initialize LiteLLM
litellm.set_verbose = False  # Set to True for debugging

# Initialize rich console
console = Console()

# Load environment variables
load_dotenv()

# Check for required environment variables
def check_environment_variables():
    required_vars = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "SERPER_API_KEY": SERPER_API_KEY,
        "MONGODB_URI": MONGODB_URI
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        exit(1)

check_environment_variables()

# Initialize LLM
def initialize_llm():
    try:
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="groq/llama-3.1-8b-instant"
        )
    except OpenAIError as e:
        logger.error(f"Failed to initialize LLM: {e.message} ([{e.status_code}]) from provider {e.llm_provider}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error initializing LLM: {str(e)}", exc_info=True)
        exit(1)

llm = initialize_llm()

# Initialize DatabaseHandler
try:
    db_handler = DatabaseHandler(MONGODB_URI)
except Exception as e:
    logger.error(f"Database connection failed: {str(e)}")
    exit(1)

# Initialize agents
def initialize_agents(llm):
    internet_search_agent = InternetSearchAgent(llm=llm, max_results=1)  # You can adjust the max_results as needed
    content_generator_agent = ContentGeneratorAgent(llm=llm)
    return internet_search_agent, content_generator_agent

agents = initialize_agents(llm)

# Create the crew with Pydantic output models
def create_recipe_crew(agents):
    return Crew(
        agents=[agent.agent for agent in agents],
        tasks=[
            Task(
                description="Search for food recipes based on user keywords: {keywords}",
                agent=agents[0].agent,
                expected_output="A list of relevant food recipes based on the provided keywords.",
                output_pydantic=SearchOutput
            ),
            Task(
                description="Generate SEO-optimized food recipe content based on search results for keywords: {keywords}",
                agent=agents[1].agent,
                expected_output="SEO-optimized content for the provided food recipes.",
                output_pydantic=ContentOutput,
                input_mapping={"search_results": "tasks_output[0].output"}
            )
        ],
        verbose=True
    )

recipe_crew = create_recipe_crew(agents)

# Initialize terminal UI
terminal_ui = TerminalUI()

def process_user_input(terminal_ui):
    while True:
        keywords = terminal_ui.get_user_input("Enter recipe keywords (or 'quit' to exit): ")
        if keywords.lower() == 'quit':
            break
        yield keywords

def parse_llm_response(response):
    try:
        # Try to parse the entire response as JSON first
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to find a JSON block
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.error("Found JSON block, but it's invalid.")
        else:
            logger.error("JSON block not found in the response.")
        
        # If all parsing attempts fail, return the raw response
        logger.warning("Returning raw response as fallback.")
        return {"raw_response": response}

def execute_crew_tasks(recipe_crew, keywords):
    logger.info("Starting recipe search and content generation...")
    try:
        result = recipe_crew.kickoff(inputs={'keywords': keywords})
        parsed_result = parse_llm_response(result.raw)
        if parsed_result is None:
            logger.error("Failed to parse LLM response. Using raw output.")
            return {'raw': result.raw, 'tasks_output': result.tasks_output}
        return {'raw': result.raw, 'tasks_output': result.tasks_output, 'parsed': parsed_result}
    except OpenAIError as e:
        logger.error(f"LiteLLM Error: {e.message} ([{e.status_code}]) from provider {e.llm_provider}")
        return None
    except Exception as e:
        logger.error(f"Unexpected Error during crew execution: {str(e)}", exc_info=True)
        return None

def validate_outputs(result):
    try:
        if isinstance(result, list) and len(result) >= 2:
            search_output = result[0].output if hasattr(result[0], 'output') else result[0]
            content_output = result[1].output if hasattr(result[1], 'output') else result[1]
        elif isinstance(result, dict) and 'tasks_output' in result:
            search_output = result['tasks_output'][0].output
            content_output = result['tasks_output'][1].output
        else:
            raise ValueError("Unexpected result structure")
        return search_output, content_output
    except (AttributeError, IndexError, KeyError) as e:
        logger.error(f"Error in output validation: {str(e)}")
        logger.debug(f"Result structure: {result}")
        return None, None
    except Exception as e:
        logger.error(f"Output validation failed: {str(e)}")
        return None, None

def save_to_mongodb(db_handler, keywords, result):
    try:
        recipe_document = {
            "keywords": keywords,
            "raw_output": result.get('raw', ''),
            "tasks_output": [
                {
                    "description": getattr(task, 'description', ''),
                    "agent": getattr(getattr(task, 'agent', None), 'name', ''),
                    "result": task.output.dict() if hasattr(task.output, 'dict') else str(task.output)
                } for task in result.get('tasks_output', [])
            ],
            "token_usage": result.get('token_usage', {})
        }
        db_handler.recipes_collection.insert_one(recipe_document)
        logger.info(f"Generated and saved recipe content for '{keywords}'")
    except Exception as e:
        logger.error(f"Failed to save to MongoDB: {str(e)}")
        logger.debug(f"Result structure: {result}")

def main():
    try:
        terminal_ui.display_welcome_message()

        for keywords in process_user_input(terminal_ui):
            result = execute_crew_tasks(recipe_crew, keywords)
            if result is None:
                continue

            logger.info(f"Raw Output: {result['raw']}")

            search_output, content_output = validate_outputs(result.get('tasks_output', []))
            if search_output is None and content_output is None:
                logger.error("Failed to validate outputs. Skipping this iteration.")
                continue

            save_to_mongodb(db_handler, keywords, result)
            terminal_ui.display_result(content_output)

        terminal_ui.display_goodbye_message()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting...")
    except OpenAIError as e:
        logger.error(f"LiteLLM Error: {e.message} ([{e.status_code}]) from provider {e.llm_provider}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
    finally:
        db_handler.close()
        logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    main()