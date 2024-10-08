from crewai import Agent
from langchain.tools import Tool
from tools.database_handler import DatabaseHandler
from rich.console import Console

console = Console()

class ScraperAgent(Agent):
    def __init__(self, llm, db_handler: DatabaseHandler):
        super().__init__(
            name="Recipe Scraper",
            role="Web Scraper",
            goal="Scrape the internet for food recipes",
            backstory="I am an expert at finding and extracting recipes from various websites.",
            llm=llm,
            tools=[
                Tool(
                    name="web_scraper",
                    func=self.scrape_recipes,
                    description="Scrape recipes from the internet"
                )
            ]
        )
        self.db_handler = db_handler

    def scrape_recipes(self, category: str):
        # Implement web scraping logic here
        # Use libraries like BeautifulSoup or Scrapy
        # This is a placeholder implementation
        console.log(f"[bold yellow]Scraping recipes for category: {category}[/bold yellow]")
        recipes = [
            {
                'title': f'Sample Recipe {i}',
                'ingredients': 'Sample ingredients',
                'instructions': 'Sample instructions',
                'category': category
            } for i in range(5)
        ]
        self.db_handler.save_recipes(recipes)
        return f"Scraped {len(recipes)} recipes for category: {category}"

    @property
    def task(self):
        return "Scrape recipes from various food websites based on given categories"