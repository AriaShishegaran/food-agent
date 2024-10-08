# Database handler implementation

from pymongo import MongoClient
from rich.console import Console

console = Console()

class DatabaseHandler:
    def __init__(self, connection_string):
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client["food_recipes"]
            self.recipes_collection = self.db["recipes"]
            self.content_collection = self.db["content"]
            console.print("[bold green]Successfully connected to MongoDB[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to connect to MongoDB: {str(e)}[/bold red]")
            raise e

    def save_recipes(self, recipes):
        try:
            self.recipes_collection.insert_many(recipes)
            console.log(f"[bold green]Saved {len(recipes)} recipes to MongoDB.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error saving recipes: {str(e)}[/bold red]")

    def save_content(self, recipe_id, content):
        try:
            self.content_collection.insert_one({
                "recipe_id": recipe_id,
                "text": content,
                "optimized": False
            })
            console.log(f"[bold green]Saved content for recipe ID {recipe_id}.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error saving content: {str(e)}[/bold red]")

    def update_content(self, content_id, optimized_content):
        try:
            self.content_collection.update_one(
                {"_id": content_id},
                {"$set": {"text": optimized_content, "optimized": True}}
            )
            console.log(f"[bold green]Updated and optimized content ID {content_id}.[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error updating content: {str(e)}[/bold red]")

    def close(self):
        self.client.close()
        console.print("[bold blue]MongoDB connection closed[/bold blue]")