from fastapi import HTTPException
from PIL import Image
from transformers import AutoModel
from deep_translator import GoogleTranslator
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import numpy as np
import ast
import json
import csv
import os
import base64
from io import BytesIO
import requests
import io
from elasticsearch import Elasticsearch

# Configuration constants

USER_EMBEDDINGS_PATH = "./user_embeddings.json"
RECIPE_EMBEDDINGS_PATH = "./embeddings.json"
DATA_PATH = "./data.csv"
FEEDBACK_PATH = "./recipe_feedback.csv"
RECIPES_ADD_PATH = "./recipes_add.csv"
REVIEWS_PATH = "./user_reviews.csv"
device = "cpu"

# Create Elasticsearch client
es = Elasticsearch(
    "http://localhost:9200",  # Changed from https to http
    basic_auth=("elastic", "pass"),  # Use your actual password
)
# Update disk watermark thresholds
es.cluster.put_settings(
    body={
        "persistent": {
            "cluster.routing.allocation.disk.watermark.low": "99%",
            "cluster.routing.allocation.disk.watermark.high": "99%",
            "cluster.routing.allocation.disk.watermark.flood_stage": "99%",
        }
    }
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


# def initialize_globals():
#     """Initialize global variables used across the application"""
#     global df, distinct_ingredients, cuisines, courses, diets
#     df = pd.read_csv(DATA_PATH)
#     print(df.head())
#     df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(ast.literal_eval)

#     # Initialize distinct values
#     distinct_ingredients = sorted(
#         list(set(ingredient for row in df["Cleaned_Ingredients"] for ingredient in row))
#     )
#     cuisines = [c for c in df["Cuisine"].dropna().unique() if c.lower() != "unknown"]
#     courses = [c for c in df["Course"].dropna().unique() if c.lower() != "unknown"]
#     diets = [d for d in df["Diet"].dropna().unique() if d.lower() != "unknown"]
#     return df, distinct_ingredients, cuisines, courses, diets


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


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def load_user_embeddings():
    if os.path.exists(USER_EMBEDDINGS_PATH):
        with open(USER_EMBEDDINGS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_user_embeddings(user_embeddings):
    try:
        with open(USER_EMBEDDINGS_PATH, "w") as f:
            json.dump(user_embeddings, f)
    except IOError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save user embeddings: {str(e)}"
        )


def get_user_embeddings(user_id, user_embeddings):
    user_id = str(user_id)
    return list(user_embeddings[user_id]) if user_id in user_embeddings else None


def update_user_embeddings(user_id, user_embeddings, new_embedding, alpha=0.8):
    user_id = str(user_id)
    if user_id in user_embeddings:
        previous_embedding = torch.tensor(user_embeddings[user_id])
        new_embedding = torch.tensor(new_embedding)
        updated_embedding = (1 - alpha) * new_embedding + alpha * previous_embedding
        user_embeddings[user_id] = [float(x) for x in updated_embedding]
    else:
        user_embeddings[user_id] = [float(x) for x in new_embedding]

    if user_embeddings:
        save_user_embeddings(user_embeddings)


def filter_df(df, **kwargs):
    filtered_df = df.copy()
    for key, value in kwargs.items():
        if key == "Title" or not value:
            continue
        if key not in df.columns:
            raise ValueError(f"Column '{key}' is not in the DataFrame.")

        if pd.api.types.is_numeric_dtype(df[key]):
            filtered_df = filtered_df[filtered_df[key] <= value]
        elif pd.api.types.is_string_dtype(df[key]):
            filtered_df = filtered_df[filtered_df[key].isin(value)]
        elif key == "Cleaned_Ingredients":
            filtered_df = filtered_df[
                filtered_df[key].apply(
                    lambda ingredients: any(
                        ingredient in ingredients for ingredient in value
                    )
                )
            ]
    return filtered_df


def compute_average_embedding(title_text=None, image=None):
    embeddings = []
    if title_text:
        result = get_embeddings([title_text])[0]
        if result.get("type") != "error":
            title_embedding = torch.tensor(result["embeddings"]).squeeze().to(device)
            embeddings.append(title_embedding)

    if image:
        # Handle image processing...
        result = get_embeddings([img_str])[0]
        if result.get("type") != "error":
            image_embedding = torch.tensor(result["embeddings"]).squeeze().to(device)
            embeddings.append(image_embedding)

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings could be generated from the input")

    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return list(avg_embedding.cpu().numpy())


# Function to find the most similar recipes
def find_most_similar_recipe(avg_embedding, df, top_n=5):
    with open(RECIPE_EMBEDDINGS_PATH, "r") as f:
        recipe_embeddings = json.load(f)

    # Convert all IDs to integers for consistency
    df_ids = set(df["ID"].astype(int))
    filtered_embeddings = {
        int(k): v for k, v in recipe_embeddings.items() if int(k) in df_ids
    }

    if not filtered_embeddings:
        return df.head(top_n)["ID"].astype(int).tolist()

    recipe_ids = list(filtered_embeddings.keys())
    embeddings = np.array(
        [np.array(embed).flatten() for embed in filtered_embeddings.values()]
    )
    avg_embedding = np.array(avg_embedding).reshape(1, -1)

    similarities = cosine_similarity(avg_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_ids = [recipe_ids[i] for i in top_indices]  # No need to convert to int again

    return top_ids


### Feedback part
import csv
from deep_translator import GoogleTranslator
from langdetect import detect


# def is_hindi(text):
#     return detect(text) == "hi"


# def translate_text(text, target_lang, source_lang="auto"):
#     translator = GoogleTranslator(source=source_lang, target=target_lang)
#     return translator.translate(text)


def update_embedding_from_feedback(user_id, title_text, image, rating):
    if not isinstance(rating, (int, float)) or rating < 0 or rating > 5:
        raise ValueError("Rating must be a number between 0 and 5")

    normalized_rating = rating / 5  # Normalize rating once
    user_embeddings = load_user_embeddings()

    if image is not None:
        try:
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            print(f"Warning: Failed to process image: {str(e)}")
            image = None

    avg_embedding = compute_average_embedding(title_text, image)
    update_user_embeddings(
        user_id,
        user_embeddings,
        new_embedding=list(avg_embedding),
        alpha=normalized_rating,
    )


def save_feedback(user_id, recipe_titles, rating, title_text, image):
    try:
        feedback_data = {
            "user_id": user_id,
            "recipe_titles": recipe_titles,
            "rating": rating,
        }

        with open(FEEDBACK_PATH, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=feedback_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(feedback_data)
        update_embedding_from_feedback(user_id, title_text, image, rating)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save feedback: {str(e)}"
        )


# Define FastAPI endpoint


# embeddings


def get_embeddings(strings_list, api_url="http://localhost:8000/encode"):
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


def save_review(review_text):
    """Save a user review to the reviews CSV file"""
    review_data = {"review_text": review_text}

    with open(REVIEWS_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=review_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(review_data)


def add_recipe(recipe_data):
    """Add a new recipe to the recipes CSV file"""
    with open(RECIPES_ADD_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=recipe_data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(recipe_data)


def predict_recipes(data, df):
    """
    Process prediction request and return recommended recipes.

    Args:
        data (dict): Request data containing user preferences and filters
        df (pd.DataFrame): Recipe dataframe

    Returns:
        tuple: (recipe_titles, details) containing recommended recipes
    """
    # Extract data from request
    user_id = data.get("user_id")
    title_text = data.get("title_text")
    prep_time = data.get("prep_time")
    cook_time = data.get("cook_time")
    selected_cuisines = data.get("selected_cuisines", [])
    selected_courses = data.get("selected_courses", [])
    selected_diets = data.get("selected_diets", [])
    selected_ingredients = data.get("selected_ingredients", [])
    image = data.get("image", None)

    # Filter the DataFrame
    filtered_df = filter_df(
        df,
        Prep_Time=prep_time,
        Cook_Time=cook_time,
        Cuisine=selected_cuisines,
        Course=selected_courses,
        Diet=selected_diets,
        Cleaned_Ingredients=selected_ingredients,
    )

    if filtered_df.empty:
        return [], "No similar recipes found"

    # Process image if provided
    if image is not None:
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))

    # Compute embeddings and get recommendations
    avg_embedding = compute_average_embedding(title_text, image)
    user_embeddings = load_user_embeddings()
    user_embedding = get_user_embeddings(user_id, user_embeddings)

    avg_embedding = np.array(avg_embedding)
    if user_embedding is not None:
        user_embedding = np.array(user_embedding)
        avg_embedding = 0.8 * avg_embedding + 0.2 * user_embedding

    # Get final recommendations
    if avg_embedding is None:
        final_df = filtered_df.head(5)
    else:
        top_ids = find_most_similar_recipe(avg_embedding, filtered_df, top_n=5)
        final_df = filtered_df[filtered_df["ID"].apply(lambda x: x in top_ids)]

    recipe_titles = final_df["Title"].tolist()
    details = final_df[
        [
            "Title",
            "Cuisine",
            "Course",
            "Diet",
            "Prep_Time",
            "Cook_Time",
            "Cleaned_Ingredients",
            "Instructions",
        ]
    ].to_markdown(index=False)

    return recipe_titles, details
