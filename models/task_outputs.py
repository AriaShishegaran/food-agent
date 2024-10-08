from pydantic import BaseModel, Field
from typing import List, Optional

class Recipe(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]
    source: str

class SearchOutput(BaseModel):
    recipes: List[Recipe]

class ContentOutput(BaseModel):
    title: str
    introduction: str
    ingredients: List[str]
    instructions: List[str]
    nutritional_info: Optional[str] = Field(default="")
    tips_and_variations: Optional[str] = Field(default="")
    conclusion: Optional[str] = Field(default="")
    seo_optimized_text: str