from crewai import Agent
import json
from models.task_outputs import ContentOutput
from rich.console import Console

console = Console()

class ContentGeneratorAgent:
    def __init__(self, llm):
        self.agent = Agent(
            name="Content Generator",
            role='Food Recipe Content Creator',
            goal='Generate SEO-optimized, beautiful food recipe content based on search results',
            backstory='I am an expert content writer specializing in food recipes, with a keen eye for SEO optimization.',
            llm=llm,
            verbose=True
        )

    def generate_content(self, search_results, keywords):
        console.log(f"[bold blue]Generating content for keywords: {keywords}[/bold blue]")
        try:
            recipes = json.loads(search_results)['recipes']
        except json.JSONDecodeError:
            recipes = [{"title": "Error", "ingredients": [], "instructions": [], "source": ""}]
            console.print("[bold red]Invalid JSON in search results. Using default error recipe.[/bold red]")

        task = f"""
        Create a beautiful, SEO-optimized food recipe article based on the following recipes for '{keywords}':
        {json.dumps(recipes, indent=2)}
        
        The article should include:
        1. An engaging title
        2. A brief introduction
        3. List of ingredients
        4. Step-by-step cooking instructions
        5. Nutritional information (if available)
        6. Tips and variations
        7. Conclusion
        
        Format the output as a JSON string with the following structure:
        {{
            "title": "Article Title",
            "introduction": "Introduction text",
            "ingredients": ["ingredient1", "ingredient2", ...],
            "instructions": ["step1", "step2", ...],
            "nutritional_info": "Nutritional information text",
            "tips_and_variations": "Tips and variations text",
            "conclusion": "Conclusion text"
        }}
        
        Ensure the content is well-structured, easy to read, and optimized for search engines. The content should be relevant to '{keywords}'.
        """
        response = self.agent.execute(task, expected_output="SEO-optimized food recipe content in JSON format.")

        # Validate the response
        try:
            content_output = ContentOutput.parse_raw(response)
            console.log(f"[bold green]Content generation successful for '{keywords}'.[/bold green]")
            return content_output
        except Exception as e:
            console.print(f"[bold red]Failed to parse content generation results: {str(e)}[/bold red]")
            raise e