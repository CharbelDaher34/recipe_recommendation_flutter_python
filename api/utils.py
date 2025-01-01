from typing import List, Optional, Tuple
import re
import json
import os
from datetime import datetime

# PIL imports
from PIL import Image

# Data processing imports
import pandas as pd
import numpy as np

# IO and encoding imports
import base64
from io import BytesIO
import requests
import io

# Elasticsearch
from elasticsearch import Elasticsearch

# Local model imports
from models import Recipe, User, Feedback, Review, RecipeAdd, UserReview

# ===========================================
# Configuration and Initialization
# ===========================================

# Configuration constants

device = "cpu"

# Create Elasticsearch client
es = Elasticsearch(
    "http://elasticsearch:9200",  # Changed from https to http
    basic_auth=("elastic", "pass"),  # Use your actual password
)

# Test connection
try:
    if es.ping():
        print("Successfully connected to Elasticsearch")
        print(es.info())
    else:
        print("Could not connect to Elasticsearch")
except Exception as e:
    print(f"Connection failed: {e}")


def initialize_globals():
    """Initialize global variables used across the application"""
    try:
        # Simple aggregation query for all fields
        query = {
            "size": 0,
            "aggs": {
                "unique_cuisines": {"terms": {"field": "cuisine", "size": 10000}},
                "unique_courses": {"terms": {"field": "course", "size": 10000}},
                "unique_diets": {"terms": {"field": "diet", "size": 10000}},
                "unique_ingredients": {
                    "terms": {"field": "ingredients.keyword", "size": 1000}
                },
            },
        }

        # Execute the search
        response = es.search(index="recipes", body=query)

        # Extract values from buckets
        cuisines = sorted(
            [
                bucket["key"]
                for bucket in response["aggregations"]["unique_cuisines"]["buckets"]
            ]
        )
        courses = sorted(
            [
                bucket["key"]
                for bucket in response["aggregations"]["unique_courses"]["buckets"]
            ]
        )
        diets = sorted(
            [
                bucket["key"]
                for bucket in response["aggregations"]["unique_diets"]["buckets"]
            ]
        )
        distinct_ingredients = sorted(
            [
                bucket["key"]
                for bucket in response["aggregations"]["unique_ingredients"]["buckets"]
            ]
        )

        print(
            f"Found {len(cuisines)} cuisines, {len(courses)} courses, {len(diets)} diets, "
            f"and {len(distinct_ingredients)} ingredients"
        )

        return distinct_ingredients, cuisines, courses, diets

    except Exception as e:
        print(f"Error initializing globals from Elasticsearch: {e}")
        return [], [], [], []


# ===========================================
# Image Processing Utilities
# ===========================================


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def is_base64_image(string):
    try:
        if not re.match("^[A-Za-z0-9+/]*={0,2}$", string):
            return False
        image = base64_to_image(string)
        return True
    except:
        return False


# ===========================================
# User Management
# ===========================================


def get_user_profile(
    email: str, es_client=es, index_name: str = "users"
) -> Optional[User]:
    """
    Retrieve a user profile from Elasticsearch by email and convert to User model

    Args:
        email: User's email address
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index (default: "users")

    Returns:
        Optional[User]: User model instance if found, None otherwise
    """
    try:
        # Validate email
        if not email or not isinstance(email, str):
            print("Invalid or empty email provided")
            return None

        # Search for user by email field instead of using it as ID
        query = {"query": {"term": {"email.keyword": email}}}

        result = es_client.search(index=index_name, body=query)

        # Check if we found any matches
        if result["hits"]["total"]["value"] == 0:
            print(f"No user found with email {email}")
            return None

        # Convert first matching document to User model
        user_data = result["hits"]["hits"][0]["_source"]
        return User(**user_data)

    except Exception as e:
        print(f"Error retrieving user profile: {e}")
        return None


def create_user(
    user: User, es_client: Elasticsearch = es, index_name: str = "users"
) -> bool:
    """
    Index a User model instance into Elasticsearch if it doesn't already exist

    Args:
        user: User model instance
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index for users (default: "users")

    Returns:
        bool: True if user was indexed successfully, False if user already exists or error occurs
    """
    try:
        # Check if user already exists
        if es_client.exists(index=index_name, id=user.email):
            print(f"User {user.email} already exists in {index_name}")
            return False

        # Convert User model to dictionary
        doc = user.model_dump()

        # Use email as document ID since it's unique
        es_client.index(index=index_name, id=user.email, document=doc)
        print(f"Successfully indexed user {user.email} to {index_name}")
        return True

    except Exception as e:
        print(f"Error indexing user: {e}")
        return False


def login_user(
    user: User, es_client: Elasticsearch = es, index_name: str = "users"
) -> bool:
    """
    Verify user credentials against Elasticsearch

    Args:
        user: User model instance containing email and password
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index for users (default: "users")

    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        # Check if user exists and get their data
        if not es_client.exists(index=index_name, id=user.email):
            print("User not found")
            return False

        # Get user data
        user_data = es_client.get(index=index_name, id=user.email)["_source"]

        # Check if password matches
        # NOTE: In production, you should use proper password hashing and verification
        if user_data["password"] == user.password:
            print("Login successful")
            return True
        else:
            print("Invalid password")
            return False

    except Exception as e:
        print(f"Error during login: {e}")
        return False


# ===========================================
# Embedding Management
# ===========================================


# def load_user_embeddings():
#     if os.path.exists(USER_EMBEDDINGS_PATH):
#         with open(USER_EMBEDDINGS_PATH, "r") as f:
#             return json.load(f)
#     return {}


# def save_user_embeddings(user_embeddings):
#     try:
#         with open(USER_EMBEDDINGS_PATH, "w") as f:
#             json.dump(user_embeddings, f)
#     except IOError as e:
#         raise HTTPException(
#             status_code=500, detail=f"Failed to save user embeddings: {str(e)}"
#         )


def index_user_embedding(
    email: str,
    embedding: List[float],
    es_client: Elasticsearch = es,
    index_name: str = "users",
) -> bool:
    """
    Update a user's embedding in Elasticsearch

    Args:
        email: User's email address (used as document ID)
        embedding: List of floats representing the user's new embedding
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index

    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        # Check if user exists
        if not es_client.exists(index=index_name, id=email):
            print(f"User {email} not found")
            return False
        # Handle nested lists by flattening if necessary
        if isinstance(embedding[0], list):
            embedding = embedding[0]
        # Ensure embedding is a list of floats
        embedding = [float(x) for x in embedding]

        # Update only the embedding field
        update_doc = {"doc": {"embedding": embedding}}

        es_client.update(index=index_name, id=email, body=update_doc)
        print(f"Successfully updated embedding for user {email}")
        return True

    except Exception as e:
        print(f"Error updating user embedding: {e}")
        return False


def update_user_embeddings(email, new_embedding, alpha=0.8):
    """
    Update a user's embedding using a weighted combination of old and new embeddings

    Args:
        email: User's email address
        new_embedding: List or array of new embedding values
        alpha: Weight for the old embedding (default: 0.8)
    """
    user = get_user_profile(email)
    new_embedding = np.array(new_embedding)

    if user is not None and user.embedding is not None:
        # Convert existing embedding to numpy array
        user_embedding = np.array(user.embedding)
        # Calculate weighted combination
        updated_embedding = (1 - alpha) * new_embedding + alpha * user_embedding
    else:
        # Use new embedding if user doesn't exist or has no embedding
        updated_embedding = new_embedding

    # Convert to list of floats and update in Elasticsearch
    updated_embedding_list = updated_embedding.tolist()
    index_user_embedding(email, updated_embedding_list)


def compute_average_embedding(title_text=None, image=None):
    embeddings = []
    if title_text:
        result = get_embeddings([title_text])[0]
        if result.get("type") != "error":
            title_embedding = np.array(result["embeddings"])
            embeddings.append(title_embedding)

    if image:
        # Convert PIL Image to base64 if needed
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            # Assume it's already base64 encoded
            img_str = image

        result = get_embeddings([img_str])[0]
        if result.get("type") != "error":
            image_embedding = np.array(result["embeddings"])
            embeddings.append(image_embedding)

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings could be generated from the input")

    # Ensure all embeddings have the same shape
    embeddings = [emb.flatten() for emb in embeddings]

    # Compute the average of all available embeddings
    avg_embedding = np.mean(embeddings, axis=0)

    return avg_embedding


def get_embeddings(strings_list, api_url="http://embedding_model:8000/encode"):
    """
    Get embeddings for a list of strings (text or base64-encoded images)

    Args:
        strings_list (list): List of strings (text or base64-encoded images)
        api_url (str): URL of the embedding API

    Returns:
        list: List of embeddings
    """
    embeddings_list = []

    for string in strings_list:
        try:
            response = requests.post(api_url, json={"content": string})
            response.raise_for_status()
            result = response.json()

            if result["status"] == "success":
                embeddings_list.append(
                    {
                        "input": (
                            string[:50] + "..." if len(string) > 50 else string
                        ),  # Truncate long strings in log
                        "type": result["type"],
                        "embeddings": result["embeddings"],
                    }
                )
            else:
                print(
                    f"Error processing string: {result.get('message', 'Unknown error')}"
                )
                embeddings_list.append(
                    {
                        "input": string[:50] + "...",
                        "type": "error",
                        "error": result.get("message", "Unknown error"),
                    }
                )

        except Exception as e:
            print(f"Error in API call: {str(e)}")
            embeddings_list.append(
                {"input": string[:50] + "...", "type": "error", "error": str(e)}
            )

    return embeddings_list


# ===========================================
# Recipe Management
# ===========================================


def filterDf(**kwargs) -> tuple[List[Recipe], pd.DataFrame]:
    """
    Create and execute an Elasticsearch query based on filter parameters

    Args:
        es_client: Elasticsearch client instance
        **kwargs: Filter parameters (prep_time, cook_time, cuisine, course, diet, ingredients)

    Returns:
        Tuple[List[Recipe], pd.DataFrame]: Tuple containing:
            - List of matching Recipe model instances
            - DataFrame containing the same data
    """
    # Start with a match_all query
    query = {"bool": {"must": [], "filter": []}}

    for key, value in kwargs.items():
        if key == "Title" or not value:
            continue

        if key in ["prep_time", "cook_time"]:
            query["bool"]["filter"].append({"range": {key.lower(): {"lte": value}}})
        elif key in ["cuisine", "course", "diet"]:
            if isinstance(value, list) and value:
                query["bool"]["must"].append({"terms": {key.lower(): value}})
        elif key == "Cleaned_Ingredients" and value:
            query["bool"]["must"].append({"terms": {"ingredients.keyword": value}})

    # Construct the full search body
    search_body = {"query": query, "size": 100}

    try:
        # Execute the search
        response = es.search(index="recipes", body=search_body)

        # Convert hits to Recipe models and collect data for DataFrame
        recipes = []
        df_data = []

        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            recipe = Recipe(
                id=source["id"],
                title=source["title"],
                ingredients=source["ingredients"],
                instructions=source["instructions"],
                prep_time=source["prep_time"],
                cook_time=source["cook_time"],
                cuisine=source["cuisine"],
                course=source["course"],
                diet=source["diet"],
                image=source.get("image"),
                url=source.get("url"),
                embedding=source.get("embedding"),
            )
            recipes.append(recipe)

            # Add the same data to the DataFrame collection
            df_data.append(
                {
                    "id": recipe.id,
                    "title": recipe.title,
                    "ingredients": recipe.ingredients,
                    "instructions": recipe.instructions,
                    "prep_time": recipe.prep_time,
                    "cook_time": recipe.cook_time,
                    "cuisine": recipe.cuisine,
                    "course": recipe.course,
                    "diet": recipe.diet,
                    "embedding": recipe.embedding,
                    "url": recipe.url,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(df_data)

        return df

    except Exception as e:
        print(f"Error executing Elasticsearch query: {e}")
        return pd.DataFrame()


def find_most_similar_recipe(
    avg_embedding, ids, top_n=5, min_similarity=0.4
) -> tuple[List[Recipe], pd.DataFrame]:
    """
    Find recipes from a specific set of IDs with embedding similarity > min_similarity

    Args:
        avg_embedding: Query embedding vector
        ids: List of recipe IDs to search within
        top_n: Maximum number of results to return (default 5)
        min_similarity: Minimum similarity threshold (default 0.7)

    Returns:
        tuple: (List[Recipe], pd.DataFrame) containing similar recipes
    """
    # Convert embedding to list if it's numpy array
    if hasattr(avg_embedding, "tolist"):
        avg_embedding = avg_embedding.tolist()

    # Query that combines ID filtering with similarity scoring
    query = {
        "query": {
            "script_score": {
                "query": {"terms": {"id": ids}},  # Filter by provided IDs
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": avg_embedding},
                },
            }
        },
        "min_score": 1 + min_similarity,  # Add 1 to adjust for the +1.0 in script
        "size": top_n,
        "_source": True,  # Get all fields
    }

    try:
        # Execute search
        response = es.search(index="recipes", body=query)

        # Convert hits to Recipe models and collect data for DataFrame
        recipes = []
        df_data = []

        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            similarity = hit["_score"] - 1.0  # Convert back to -1 to 1 range

            # Create Recipe model
            recipe = Recipe(
                id=source["id"],
                title=source["title"],
                ingredients=source["ingredients"],
                instructions=source["instructions"],
                prep_time=source["prep_time"],
                cook_time=source["cook_time"],
                cuisine=source["cuisine"],
                course=source["course"],
                diet=source["diet"],
                image=source.get("image"),
                url=source.get("url"),
                embedding=source.get("embedding"),
            )
            recipes.append(recipe)

            # Add data for DataFrame
            df_data.append(
                {
                    "id": recipe.id,
                    "title": recipe.title,
                    "ingredients": recipe.ingredients,
                    "instructions": recipe.instructions,
                    "prep_time": recipe.prep_time,
                    "cook_time": recipe.cook_time,
                    "cuisine": recipe.cuisine,
                    "course": recipe.course,
                    "diet": recipe.diet,
                    "embedding": recipe.embedding,
                    "url": recipe.url,
                    "similarity": similarity,
                }
            )

        # Create DataFrame
        df = pd.DataFrame(df_data)

        return df

    except Exception as e:
        print(f"Error finding similar recipes: {e}")
        return pd.DataFrame(
            columns=[
                "id",
                "title",
                "ingredients",
                "instructions",
                "prep_time",
                "cook_time",
                "cuisine",
                "course",
                "diet",
                "embedding",
                "url",
                "similarity",
            ]
        )


def predict_recipes(data):
    """
    Process prediction request and return recommended recipes.

    Args:
        data (dict): Request data containing user preferences and filters

    Returns:
        tuple: (recipe_titles, details) containing recommended recipes
    """
    # Extract data from request
    email = data.get("email")
    title_text = data.get("title_text")
    prep_time = data.get("prep_time")
    cook_time = data.get("cook_time")
    selected_cuisines = data.get("selected_cuisines", [])
    selected_courses = data.get("selected_courses", [])
    selected_diets = data.get("selected_diets", [])
    selected_ingredients = data.get("selected_ingredients", [])
    image = data.get("image", None)
    print(
        prep_time,
        cook_time,
        selected_cuisines,
        selected_courses,
        selected_diets,
        selected_ingredients,
    )
    # Filter the DataFrame
    filtered_df = filterDf(
        Prep_Time=prep_time,
        Cook_Time=cook_time,
        Cuisine=selected_cuisines,
        Course=selected_courses,
        Diet=selected_diets,
        Cleaned_Ingredients=selected_ingredients,
    )
    if filtered_df.empty:
        return [], "No similar recipes found"

    # Compute embeddings and get recommendations
    avg_embedding = compute_average_embedding(title_text, image)

    # Get user embedding from Elasticsearch
    user = get_user_profile(email)
    if user and user.embedding is not None:
        user_embedding = np.array(user.embedding)
        avg_embedding = np.array(avg_embedding)
        # Combine user preferences with current query
        avg_embedding = 0.8 * avg_embedding + 0.2 * user_embedding
    # Get final recommendations
    if avg_embedding is None:
        return filtered_df.head(5)
    else:
        ids = filtered_df["id"]
        return find_most_similar_recipe(
            avg_embedding, ids, top_n=100, min_similarity=0.3
        )


def add_recipe(
    recipe: Recipe, es_client: Elasticsearch = es, index_name: str = "recipe_additions"
) -> None:
    """
    Convert Recipe to RecipeAdd and index it to Elasticsearch with accepted=False

    Args:
        recipe: Recipe model instance
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index for pending recipes
    """
    # Convert to pending RecipeAdd
    recipe_dict = recipe.model_dump()
    recipe_dict["accepted"] = False

    # Create new RecipeAdd instance
    pending_recipe = RecipeAdd(**recipe_dict)
    # Prepare document
    doc = pending_recipe.model_dump()

    try:
        # Index the document
        es_client.index(index=index_name, id=str(recipe.id), document=doc)
        print(f"Successfully indexed pending recipe {recipe.id} to {index_name}")
    except Exception as e:
        print(f"Error indexing pending recipe: {e}")


# ===========================================
# Feedback and Reviews
# ===========================================


def update_embedding_from_feedback(email, description, image, rating):
    if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
        raise ValueError("Rating must be a number between 0 and 5")

    normalized_rating = rating / 5  # Normalize rating once

    if image is not None:
        try:
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            print(f"Warning: Failed to process image: {str(e)}")
            image = None

    avg_embedding = compute_average_embedding(description, image)
    update_user_embeddings(
        email,
        new_embedding=list(avg_embedding),
        alpha=normalized_rating,
    )


def index_feedback(
    feedback: Feedback, es_client: Elasticsearch = es, index_name: str = "feedback"
) -> bool:
    """
    Index a Feedback model instance into Elasticsearch

    Args:
        feedback: Feedback model instance
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index (default: "feedback")

    Returns:
        bool: True if feedback was indexed successfully, False if error occurs
    """
    try:
        # Convert Feedback model to dictionary
        doc = feedback.model_dump()

        # Generate a unique ID (you might want to use a different strategy)
        doc_id = f"{feedback.email}_{feedback.created_at}"

        # Index the document
        es_client.index(index=index_name, id=doc_id, document=doc)
        print(
            f"Successfully indexed feedback from {feedback.email} at {feedback.created_at}"
        )
        update_embedding_from_feedback(
            feedback.email,
            feedback.input_description,
            feedback.input_image,
            feedback.rating,
        )

        return True

    except Exception as e:
        print(f"Error indexing feedback: {e}")
        return False


def save_review(
    email: str,
    review: Review,  # Changed from review_text: str
    es_client: Elasticsearch = es,
    index_name: str = "user_reviews",
) -> bool:
    """
    Save a user review to Elasticsearch. If the user already has reviews, append to their list.
    If not, create a new document for the user.

    Args:
        email: User's email address
        review: Review object containing content and creation date
        es_client: Elasticsearch client instance
        index_name: Name of the Elasticsearch index (default: "user_reviews")

    Returns:
        bool: True if review was saved successfully, False otherwise
    """
    try:
        # Check if user already has reviews
        if es_client.exists(index=index_name, id=email):
            # Get existing reviews
            result = es_client.get(index=index_name, id=email)
            user_reviews = result["_source"]["reviews"]

            # Append new review
            user_reviews.append(review.model_dump())

            # Update document
            update_doc = {"doc": {"reviews": user_reviews}}
            es_client.update(index=index_name, id=email, body=update_doc)
        else:
            # Create new UserReview document
            user_review = UserReview(email=email, reviews=[review])
            # Index new document
            es_client.index(
                index=index_name, id=email, document=user_review.model_dump()
            )

        print(f"Successfully saved review for user {email}")
        return True

    except Exception as e:
        print(f"Error saving review: {e}")
        return False
