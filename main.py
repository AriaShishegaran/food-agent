import os
import traceback
import logging
from typing import List, Dict, Any, Optional, Tuple
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

def check_environment_variables() -> None:
    """
    Check for required environment variables and exit if any are missing.
    """
    required_vars = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "SERPER_API_KEY": SERPER_API_KEY,
        "MONGODB_URI": MONGODB_URI
    }
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        exit(1)

def initialize_llm() -> ChatGroq:
    """
    Initialize and return the Language Model.

    Returns:
        ChatGroq: Initialized Language Model.

    Raises:
        SystemExit: If LLM initialization fails.
    """
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

def initialize_agents(llm: ChatGroq) -> List[Any]:
    """
    Initialize and return the agents.

    Args:
        llm (ChatGroq): The Language Model to use for the agents.

    Returns:
        List[Any]: List of initialized agents.
    """
    internet_search_agent = InternetSearchAgent(llm=llm, max_results=1)
    content_generator_agent = ContentGeneratorAgent(llm=llm)
    return [internet_search_agent, content_generator_agent]

def create_recipe_crew(agents: List[Any]) -> Crew:
    """
    Create and return the recipe crew.

    Args:
        agents (List[Any]): List of agents to include in the crew.

    Returns:
        Crew: The created recipe crew.
    """
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

def process_user_input(terminal_ui: TerminalUI) -> str:
    """
    Process user input for recipe keywords.

    Args:
        terminal_ui (TerminalUI): The terminal UI object.

    Yields:
        str: User input keywords.
    """
    while True:
        keywords = terminal_ui.get_user_input("Enter recipe keywords (or 'quit' to exit): ")
        if keywords.lower() == 'quit':
            break
        yield keywords

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM response and extract JSON data.

    Args:
        response (str): The raw LLM response.

    Returns:
        Dict[str, Any]: Parsed JSON data or raw response.
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.error("Found JSON block, but it's invalid.")
        else:
            logger.error("JSON block not found in the response.")
        
        logger.warning("Returning raw response as fallback.")
        return {"raw_response": response}

def execute_crew_tasks(recipe_crew: Crew, keywords: str) -> Optional[Dict[str, Any]]:
    """
    Execute crew tasks for recipe search and content generation.

    Args:
        recipe_crew (Crew): The recipe crew object.
        keywords (str): The search keywords.

    Returns:
        Optional[Dict[str, Any]]: The execution result or None if an error occurred.
    """
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

def validate_outputs(result: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Validate the outputs from the crew execution.

    Args:
        result (Any): The result from crew execution.

    Returns:
        Tuple[Optional[Any], Optional[Any]]: Validated search and content outputs.
    """
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

def save_to_mongodb(db_handler: DatabaseHandler, keywords: str, result: Dict[str, Any]) -> None:
    """
    Save the generated recipe content to MongoDB.

    Args:
        db_handler (DatabaseHandler): The database handler object.
        keywords (str): The search keywords.
        result (Dict[str, Any]): The result to be saved.
    """
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

def main() -> None:
    """
    Main function to run the recipe content generation process.
    """
    try:
        check_environment_variables()
        llm = initialize_llm()
        db_handler = DatabaseHandler(MONGODB_URI)
        agents = initialize_agents(llm)
        recipe_crew = create_recipe_crew(agents)
        terminal_ui = TerminalUI()

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
