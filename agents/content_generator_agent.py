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
        task = f"""
        Your task is to create an SEO-optimized food recipe article based on the following recipes for '{keywords}':
        {json.dumps(search_results.dict(), indent=2)}

        Follow these steps:
        1. Analyze the provided recipes and create a unified article.
        2. Format your response as a JSON string with the following structure:
        {{
            "title": "Article Title",
            "introduction": "Introduction text",
            "ingredients": ["ingredient1", "ingredient2", ...],
            "instructions": ["step1", "step2", ...],
            "nutritional_info": "Nutritional information text",
            "tips_and_variations": "Tips and variations text",
            "conclusion": "Conclusion text",
            "seo_optimized_text": "SEO-optimized meta description"
        }}
        3. Ensure the content is well-structured, easy to read, and optimized for search engines.
        4. The content should be relevant to '{keywords}'.
        5. Do not include any additional text or explanations outside of the JSON structure.
        """
        response = self.agent.execute(task)

        try:
            content_data = json.loads(response)
            content_output = ContentOutput(**content_data)
            console.log(f"[bold green]Content generation successful for '{keywords}'.[/bold green]")
            return content_output
        except json.JSONDecodeError as e:
            console.print(f"[bold red]Failed to parse JSON: {str(e)}[/bold red]")
            console.print(f"Raw response: {response}")
            raise
        except Exception as e:
            console.print(f"[bold red]Failed to process content generation results: {str(e)}[/bold red]")
            raise