from typing import Any
from crewai import Agent
from crewai_tools import SerperDevTool
from config.settings import SERPER_API_KEY
from models.task_outputs import SearchOutput, Recipe
from rich.console import Console
import json

console = Console()

class InternetSearchAgent:
    """
    Agent responsible for searching the internet for recipes based on user keywords.
    """

    def __init__(self, llm: Any, max_results: int = 1):
        """
        Initialize the InternetSearchAgent.

        Args:
            llm (Any): The language model to use for the agent.
            max_results (int): Maximum number of recipes to return.
        """
        self.search_tool = SerperDevTool(api_key=SERPER_API_KEY)
        self.max_results = max_results
        self.agent = Agent(
            name="Internet Researcher",
            role='Internet Researcher',
            goal='Find relevant food recipes based on user keywords',
            backstory='I am an expert at searching the internet for specific food recipes and extracting relevant information.',
            tools=[self.search_tool],
            llm=llm,
            verbose=True
        )

    def search_recipes(self, keywords: str) -> SearchOutput:
        """
        Search for recipes based on the given keywords.

        Args:
            keywords (str): The keywords to search for.

        Returns:
            SearchOutput: The search results containing recipes.
        """
        console.log(f"[bold blue]Searching for recipes with keywords: {keywords}[/bold blue]")
        task = f"""
        Your task is to search for recipes related to '{keywords}'. Follow these steps precisely:
        1. Use the search tool to find relevant recipes.
        2. Extract information for the top {self.max_results} recipe(s).
        3. For each recipe, extract the title, ingredients, instructions, and source URL.
        4. Format the information exactly as shown below:

        Thought: [Your thought process here]
        Action: [The action you are taking]
        Action Input: [Input for the action]
        ```

        ```json
        {{
            "recipes": [
                {{
                    "title": "Recipe Title",
                    "ingredients": ["ingredient1", "ingredient2", ...],
                    "instructions": ["step1", "step2", ...],
                    "source": "URL of the recipe"
                }}
            ]
        }}
        ```
        
        5. Ensure the JSON is valid and adhere strictly to the structure.
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
            return SearchOutput(recipes=[])
        except Exception as e:
            console.print(f"[bold red]Failed to process search results: {str(e)}[/bold red]")
            return SearchOutput(recipes=[])