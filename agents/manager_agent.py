from crewai import Agent

class ManagerAgent:
    def __init__(self, llm, db_manager):
        self.agent = Agent(
            role='Project Manager',
            goal='Coordinate tasks and ensure efficient workflow',
            backstory='Experienced project manager with a knack for optimization',
            llm=llm,
            verbose=True
        )
        self.db_manager = db_manager  # Keep this separate from Pydantic models

    def coordinate_tasks(self, task_info):
        # Implement task coordination logic here
        return f"Coordinated task: {task_info}"

    @property
    def task(self):
        return "Coordinate the workflow, assign tasks, and ensure non-repetitive work"