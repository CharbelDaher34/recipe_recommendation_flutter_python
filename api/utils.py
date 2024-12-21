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

# Configuration constants

USER_EMBEDDINGS_PATH = "./user_embeddings.json"
RECIPE_EMBEDDINGS_PATH = "./embeddings.json"
DATA_PATH = "./data.csv"
FEEDBACK_PATH = "./recipe_feedback.csv"
RECIPES_ADD_PATH = "./recipes_add.csv"
REVIEWS_PATH = "./user_reviews.csv"
device = "cpu"


def initialize_globals():
    """Initialize global variables used across the application"""
    global df, distinct_ingredients, cuisines, courses, diets
    df = pd.read_csv(DATA_PATH)
    print(df.head())
    df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(ast.literal_eval)

    # Initialize distinct values
    distinct_ingredients = sorted(
        list(set(ingredient for row in df["Cleaned_Ingredients"] for ingredient in row))
    )
    cuisines = [c for c in df["Cuisine"].dropna().unique() if c.lower() != "unknown"]
    courses = [c for c in df["Course"].dropna().unique() if c.lower() != "unknown"]
    diets = [d for d in df["Diet"].dropna().unique() if d.lower() != "unknown"]
    return df, distinct_ingredients, cuisines, courses, diets


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
    with open(USER_EMBEDDINGS_PATH, "w") as f:
        json.dump(user_embeddings, f)


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
        # Convert text to embedding using the API
        result = get_embeddings([title_text])[0]
        if result.get("type") != "error":
            title_embedding = torch.tensor(result["embeddings"]).to(device)
            embeddings.append(title_embedding)

    if image:
        # Handle different image input types
        if isinstance(image, str):
            # If image is a file path
            try:
                image = Image.open(image)
                img_str = image_to_base64(image)
            except Exception as e:
                # If not a path, assume it's already a base64 string
                img_str = image
        elif isinstance(image, Image.Image):
            # If image is already a PIL Image
            img_str = image_to_base64(image)
        elif hasattr(image, "read"):
            # If image is a file-like object
            image = Image.open(image)
            img_str = image_to_base64(image)
        else:
            # Assume it's already a base64 string
            img_str = image

        # Get embedding from API
        result = get_embeddings([img_str])[0]
        if result.get("type") != "error":
            image_embedding = torch.tensor(result["embeddings"]).to(device)
            embeddings.append(image_embedding)

    if len(embeddings) == 0:
        return list(torch.zeros(768).cpu().numpy())

    avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
    return list(avg_embedding.cpu().numpy())


# Function to find the most similar recipes
def find_most_similar_recipe(avg_embedding, df, top_n=5):
    # Use RECIPE_EMBEDDINGS_PATH instead of embeddings_json_path parameter
    with open(RECIPE_EMBEDDINGS_PATH, "r") as f:
        recipe_embeddings = json.load(f)

    # Rest of the function remains the same
    df_ids = set(df["ID"].astype(str))
    filtered_embeddings = {k: v for k, v in recipe_embeddings.items() if k in df_ids}
    recipe_ids = list(filtered_embeddings.keys())
    embeddings = [torch.tensor(embed) for embed in filtered_embeddings.values()]
    similarities = cosine_similarity([avg_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    top_ids = [int(recipe_ids[i]) for i in top_indices]
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
    user_embeddings = load_user_embeddings()
    if image is not None:
        # read the image
        try:
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            try:
                image = Image.fromarray(np.array(image))
            except Exception as e:
                pass
    # Compute average embedding from title and/or image
    # input_is_hindi = is_hindi(title_text) if title_text else False
    # if input_is_hindi:
    #     title_text = translate_text(title_text, "en", "hi")
    avg_embedding = compute_average_embedding(title_text, image)
    update_user_embeddings(
        user_id, user_embeddings, new_embedding=list(avg_embedding), alpha=rating / 5
    )


def save_feedback(user_id, recipe_titles, rating, title_text, image):
    # Use FEEDBACK_PATH instead of hardcoded path
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
    update_embedding_from_feedback(user_id, title_text, image, rating / 5)


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
