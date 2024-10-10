import os
import traceback
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

# Install rich traceback handler
install(show_locals=True)

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
    for var_name, var_value in required_vars.items():
        if not var_value:
            console.print(f"[bold red]Error: {var_name} is not set in the environment variables.[/bold red]")
            exit(1)

check_environment_variables()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="groq/llama-3.2-1b-preview")

# Add this near the top of the file, after imports
litellm.set_verbose = True

# Initialize DatabaseHandler
db_handler = DatabaseHandler(MONGODB_URI)

# Initialize agents
def initialize_agents(llm):
    internet_search_agent = InternetSearchAgent(llm=llm)
    content_generator_agent = ContentGeneratorAgent(llm=llm)
    return internet_search_agent, content_generator_agent

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

# Initialize terminal UI
terminal_ui = TerminalUI()

def process_user_input(terminal_ui):
    while True:
        keywords = terminal_ui.get_user_input("Enter recipe keywords (or 'quit' to exit): ")
        if keywords.lower() == 'quit':
            break
        yield keywords

def execute_crew_tasks(recipe_crew, keywords):
    console.print("[bold green]Starting recipe search and content generation...[/bold green]")
    try:
        result = recipe_crew.kickoff(inputs={'keywords': keywords})
        return result
    except Exception as e:
        console.print(f"[bold red]Error during crew execution: {str(e)}[/bold red]")
        console.print(traceback.format_exc())
        return None

def validate_outputs(result):
    try:
        search_output = result.tasks_output[0].output
        content_output = result.tasks_output[1].output
        return search_output, content_output
    except AttributeError as e:
        console.print(f"[bold red]Attribute error in output validation: {str(e)}[/bold red]")
        console.print(f"Task output structure: {result.tasks_output}")
        return None, None
    except Exception as e:
        console.print(f"[bold red]Output validation failed: {str(e)}[/bold red]")
        return None, None

def save_to_mongodb(db_handler, keywords, result):
    try:
        recipe_document = {
            "keywords": keywords,
            "raw_output": result.raw,
            "tasks_output": [
                {
                    "description": task.description,
                    "agent": task.agent.name,
                    "result": task.output.dict() if hasattr(task.output, 'dict') else task.output
                } for task in result.tasks_output
            ],
            "token_usage": result.token_usage.dict() if result.token_usage else {}
        }
        db_handler.recipes_collection.insert_one(recipe_document)
        console.print(f"[bold green]Generated and saved recipe content for '{keywords}'[/bold green]")
    except AttributeError as e:
        console.print(f"[bold red]Attribute error while saving to MongoDB: {str(e)}[/bold red]")
        console.print(f"Result structure: {result}")
    except Exception as e:
        console.print(f"[bold red]Failed to save to MongoDB: {str(e)}[/bold red]")

def main():
    try:
        load_dotenv()
        check_environment_variables()

        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="groq/llama-3.2-1b-preview")
        db_handler = DatabaseHandler(MONGODB_URI)
        agents = initialize_agents(llm)
        recipe_crew = create_recipe_crew(agents)
        terminal_ui = TerminalUI()

        terminal_ui.display_welcome_message()

        for keywords in process_user_input(terminal_ui):
            result = execute_crew_tasks(recipe_crew, keywords)
            if result is None:
                continue

            console.log(f"[bold blue]Raw Output: {result.raw}[/bold blue]")

            search_output, content_output = validate_outputs(result)
            if search_output is None or content_output is None:
                console.print("[bold red]Failed to validate outputs. Skipping this iteration.[/bold red]")
                continue

            save_to_mongodb(db_handler, keywords, result)
            terminal_ui.display_result(content_output)

        terminal_ui.display_goodbye_message()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process interrupted by user. Exiting...[/bold yellow]")
    except Exception as e:
        console.print_exception(show_locals=True)
    finally:
        db_handler.close()

if __name__ == "__main__":
    main()