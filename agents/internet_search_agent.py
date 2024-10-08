from crewai import Agent
from crewai_tools import SerperDevTool
from config.settings import SERPER_API_KEY
from models.task_outputs import SearchOutput
from rich.console import Console
import json

console = Console()

class InternetSearchAgent:
    def __init__(self, llm):
        self.search_tool = SerperDevTool(api_key=SERPER_API_KEY)
        self.agent = Agent(
            name="Internet Researcher",
            role='Internet Researcher',
            goal='Find relevant food recipes based on user keywords',
            backstory='I am an expert at searching the internet for specific food recipes and extracting relevant information.',
            tools=[self.search_tool],
            llm=llm,
            verbose=True
        )

    def search_recipes(self, keywords):
        console.log(f"[bold blue]Searching for recipes with keywords: {keywords}[/bold blue]")
        task = f"""
        Perform a detailed search for authentic '{keywords}' recipes, focusing on traditional methods and ingredients.
        Format the output as a JSON string with the following structure:
        {{
            "recipes": [
                {{
                    "title": "Recipe Title",
                    "ingredients": ["ingredient1", "ingredient2", ...],
                    "instructions": ["step1", "step2", ...],
                    "source": "URL of the recipe"
                }},
                ...
            ]
        }}
        Ensure that the recipes are well-reviewed and come from recognized cooking platforms or food blogs.
        """
        response = self.agent.execute(task, expected_output="A list of relevant food recipes in JSON format.")
        
        # Validate the response
        try:
            search_output = SearchOutput.parse_raw(response)
            if not search_output.recipes:
                raise ValueError("No recipes found.")
            console.log(f"[bold green]Successfully fetched {len(search_output.recipes)} recipes.[/bold green]")
            return search_output
        except Exception as e:
            console.print(f"[bold red]Failed to parse search results: {str(e)}[/bold red]")
            raise e