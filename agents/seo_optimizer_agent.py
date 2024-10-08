from crewai import Agent
from langchain.tools import Tool
from rich.console import Console

console = Console()

class SEOOptimizerAgent(Agent):
    def __init__(self, llm, db_manager):
        super().__init__(
            name="SEO Optimizer",
            role="SEO Specialist",
            goal="Optimize content for maximum SEO adherence",
            backstory="I am an SEO expert with years of experience in optimizing food-related content.",
            llm=llm,
            tools=[
                Tool(
                    name="seo_optimizer",
                    func=self.optimize_content,
                    description="Optimize content for SEO"
                )
            ]
        )
        self.db_manager = db_manager

    def optimize_content(self, content_id, content):
        # Implement SEO optimization logic here
        console.log(f"[bold yellow]Optimizing content ID: {content_id}[/bold yellow]")
        optimized_content = f"SEO optimized: {content}"
        self.db_manager.update_content(content_id, optimized_content)
        return optimized_content

    @property
    def task(self):
        return "Optimize the generated content for maximum SEO adherence"