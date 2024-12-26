from pydantic import BaseModel, EmailStr, HttpUrl, Field
from typing import Optional, List
from datetime import datetime


class User(BaseModel):
    email: EmailStr
    name: str
    password: str
    embedding: Optional[List[float]] = None


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
    input_description: Optional[str] = None
    input_image: Optional[str] = None
    recipe_ids: List[int]
    rating: int
    comment: str
    created_at: datetime = Field(default_factory=datetime.now)


class Review(BaseModel):
    content: str
    created_at: datetime = Field(default_factory=datetime.now)


class UserReview(BaseModel):
    email: EmailStr
    reviews: List[Review]


class RecipeAdd(Recipe):
    accepted: bool
