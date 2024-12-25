from pydantic import BaseModel, EmailStr, HttpUrl
from typing import Optional, List


class User(BaseModel):
    email: EmailStr
    name: str
    password: str
    embedding: List[float] = None


class Recipe(BaseModel):
    id: int
    title: str
    ingredients: List[str]
    instructions: List[str]
    prep_time: int  # in minutes
    cook_time: int  # in minutes
    cuisine: str
    course: str
    diet: str
    image: Optional[str] = None
    url: Optional[str] = None
    embedding: Optional[List[float]] = None


class Feedback(BaseModel):
    email: EmailStr
    input_description: str = None
    input_image: str = None
    recipe_id: int
    rating: int
    comment: str


class UserReview(BaseModel):
    email: EmailStr
    reviews: List[str]


class RecipeAdd(Recipe):
    accepted: bool
