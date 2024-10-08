from pydantic import BaseModel
from typing import List, Optional
from pymongo.collection import Collection  # Import only if necessary

class SearchOutput(BaseModel):
    recipes: List[dict]

class ContentOutput(BaseModel):
    title: str
    introduction: str
    ingredients: List[str]
    instructions: List[str]
    nutritional_info: str
    tips_and_variations: str
    conclusion: str
    seo_optimized_text: str