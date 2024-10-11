import rich
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from crewai.crews import CrewOutput
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
import time
from models.task_outputs import ContentOutput
from rich.markdown import Markdown

class TerminalUI:
    def __init__(self):
        self.console = Console()
    
    def display_welcome_message(self):
        self.console.print("╭─────────────────────────────── Recipe Content Generator ──────────────╮", style="bold green")
        self.console.print("│ Welcome to the AI-powered Recipe Content Generator!                             │", style="bold green")
        self.console.print("╰────────────────────────────────────────────────────────────────────────────╯", style="bold green")

    def get_user_input(self, prompt):
        return self.console.input(f"[bold cyan]{prompt}[/bold cyan]")

    def display_result(self, content):
        if isinstance(content, ContentOutput):
            markdown_content = f"""# {content.title}

## Introduction
{content.introduction}

## Ingredients
{content.ingredients}

## Instructions
{content.instructions}

## SEO Optimization
{content.seo_optimized_text}
"""
        else:
            # Fallback for TaskOutput or other types
            markdown_content = f"""# Recipe Result

{str(content)}
"""
        self.console.print(Markdown(markdown_content))

    def display_agent_status(self, agent_name, status):
        status_text = Text()
        status_text.append(f"{agent_name}: ", style="bold cyan")
        status_text.append(status, style="yellow")
        self.console.print(status_text)

    def display_task_progress(self, task_description):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(task_description, total=None)
            while not progress.finished:
                progress.update(task, advance=0.1)
                time.sleep(0.1)

    def display_crew_summary(self, crew):
        table = Table(title="Crew Summary")
        table.add_column("Agent", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Goal", style="green")

        for agent in crew.agents:
            table.add_row(agent.name, agent.role, agent.goal)

        self.console.print(table)

    def display_goodbye_message(self):
        self.console.print("\nThank you for using the Recipe Content Generator. Goodbye!", style="bold green")