from crewai import Agent
from crewai_tools import SerperDevTool
from config.settings import SERPER_API_KEY
from models.task_outputs import SearchOutput, Recipe
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
        Your task is to search for recipes related to '{keywords}'. Follow these steps:
        1. Use the search tool to find relevant recipes.
        2. For each recipe, extract the title, ingredients, instructions, and source URL.
        3. Format the information as a JSON string with the following structure:
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
        4. Include at least 3 recipes in your response.
        5. Ensure all recipes are well-reviewed and from reputable sources.
        6. Do not include any additional text or explanations outside of the JSON structure.
        """
        response = self.agent.execute(task)
        
        try:
            recipes_data = json.loads(response)
            search_output = SearchOutput(**recipes_data)
            console.log(f"[bold green]Successfully fetched {len(search_output.recipes)} recipes.[/bold green]")
            return search_output
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Failed to parse JSON: {str(e)}[/bold red]")
            console.print(f"Raw response: {response}")
            # Return a default SearchOutput with an empty list of recipes
            return SearchOutput(recipes=[])
        except Exception as e:
            console.print(f"[bold red]Failed to process search results: {str(e)}[/bold red]")
            # Return a default SearchOutput with an empty list of recipes
            return SearchOutput(recipes=[])