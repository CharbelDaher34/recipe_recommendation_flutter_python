# Standard library imports
import ast

# Third-party imports
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pandas as pd
import numpy as np

# Elasticsearch mappings
MAPPINGS = {
    "users": {
        "mappings": {
            "properties": {
                "email": {"type": "keyword"},
                "name": {"type": "text"},
                "password": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                },
            }
        }
    },
    "recipes": {
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "title": {"type": "text"},
                "ingredients": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "instructions": {"type": "text"},
                "prep_time": {"type": "integer"},
                "cook_time": {"type": "integer"},
                "cuisine": {"type": "keyword"},
                "course": {"type": "keyword"},
                "diet": {"type": "keyword"},
                "image": {"type": "keyword", "index": False},
                "url": {"type": "keyword", "index": False},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "similarity": "cosine",
                },
            }
        }
    },
    "feedback": {
        "mappings": {
            "properties": {
                "email": {"type": "keyword"},
                "input_description": {"type": "text"},
                "input_image": {"type": "text", "index": False},
                "recipe_ids": {"type": "integer"},
                "rating": {"type": "integer"},
                "comment": {"type": "text"},
                "created_at": {"type": "date"},
            }
        }
    },
    "user_reviews": {
        "mappings": {
            "properties": {
                "email": {"type": "keyword"},
                "reviews": {
                    "type": "nested",
                    "properties": {
                        "content": {"type": "text"},
                        "created_at": {"type": "date"},
                    },
                },
            }
        }
    },
    "recipe_additions": {
        "mappings": {
            "properties": {
                "id": {"type": "integer"},
                "title": {"type": "text"},
                "ingredients": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "instructions": {"type": "text"},
                "prep_time": {"type": "integer"},
                "cook_time": {"type": "integer"},
                "cuisine": {"type": "keyword"},
                "course": {"type": "keyword"},
                "diet": {"type": "keyword"},
                "image": {"type": "keyword", "index": False},
                "url": {"type": "keyword", "index": False},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                },
                "accepted": {"type": "boolean"},
            }
        }
    },
}


# def create_elasticsearch_client():
#     """Create and configure Elasticsearch client"""
#     es = Elasticsearch(
#         "http://elasticsearch:9200",
#         basic_auth=("elastic", "pass"),
#     )

#     # Update disk watermark thresholds
#     es.cluster.put_settings(
#         body={
#             "persistent": {
#                 "cluster.routing.allocation.disk.watermark.low": "99%",
#                 "cluster.routing.allocation.disk.watermark.high": "99%",
#                 "cluster.routing.allocation.disk.watermark.flood_stage": "99%",
#             }
#         }
#     )

#     return es
# from elasticsearch import Elasticsearch


def create_elasticsearch_client():
    """Create and configure Elasticsearch client"""
    es = Elasticsearch(
        "http://elasticsearch:9200",
        basic_auth=("elastic", "pass"),
    )

    try:
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
        print("Disk watermark thresholds updated successfully.")

        # Set default replicas for new indices
        es.indices.put_settings(
            index="_all",  # Apply to all indices
            body={"settings": {"index.number_of_replicas": 0}},
        )
        print("Default index replicas set to 0.")

    except Exception as e:
        print(f"Error during Elasticsearch configuration: {e}")

    return es


def create_indices(es_client):
    """Create all required indices if they don't exist"""
    for index_name, mapping in MAPPINGS.items():
        try:
            if not es_client.indices.exists(index=index_name):
                print(f"Creating index '{index_name}'...")
                es_client.indices.create(index=index_name, body=mapping)
                print(f"Successfully created index '{index_name}'")
            else:
                print(f"Index '{index_name}' already exists")
        except Exception as e:
            print(f"Error creating index '{index_name}': {e}")


def row_to_recipe(row):
    """Convert a DataFrame row to a Recipe object"""
    embedding = np.array(row.embedding).flatten().tolist()
    try:
        return Recipe(
            id=row.id,
            title=row.title,
            ingredients=(
                ast.literal_eval(row.ingredients)
                if isinstance(row.ingredients, str)
                else row.ingredients
            ),
            instructions=(
                ast.literal_eval(row.instructions)
                if isinstance(row.instructions, str)
                else row.instructions
            ),
            prep_time=row.prep_time,
            cook_time=row.cook_time,
            cuisine=row.cuisine,
            course=row.course,
            diet=row.diet,
            image=row.image if pd.notna(row.image) else None,
            url=row.url if pd.notna(row.url) else None,
            embedding=embedding,
        )
    except Exception as e:
        print(f"Error converting row to recipe: {e}")
        return None


def bulk_index_recipe_batch(df_batch, es_client, index_name="recipes"):
    """Convert a batch of DataFrame rows to Recipe objects and bulk index them"""
    recipes = [
        r
        for r in (row_to_recipe(row) for _, row in df_batch.iterrows())
        if r is not None
    ]

    if not recipes:
        print("No valid recipes in this batch")
        return

    actions = []
    for recipe in recipes:
        doc = {
            "id": recipe.id,
            "title": recipe.title,
            "ingredients": recipe.ingredients,
            "instructions": recipe.instructions,
            "prep_time": recipe.prep_time,
            "cook_time": recipe.cook_time,
            "cuisine": recipe.cuisine,
            "course": recipe.course,
            "diet": recipe.diet,
        }

        if recipe.image is not None:
            doc["image"] = recipe.image
        if recipe.url is not None:
            doc["url"] = recipe.url
        if recipe.embedding is not None:
            doc["embedding"] = recipe.embedding

        actions.append({"_index": index_name, "_id": str(recipe.id), "_source": doc})

    try:
        success, failed = bulk(es_client, actions, chunk_size=500, request_timeout=30)
        print(f"Successfully indexed {success} documents")
        if failed:
            print(f"Failed to index {len(failed)} documents")
    except Exception as e:
        print(f"Error during bulk indexing: {e}")


def index_recipes_data(es_client, data_path="./data.csv", batch_size=1000):
    """Read and index recipes data from CSV file"""
    try:
        # Read CSV file
        print(f"Reading data from {data_path}...")
        df = pd.read_csv(data_path)

        # Convert embedding strings to lists
        df["embedding"] = df["embedding"].apply(ast.literal_eval)

        # Process in batches
        print("Starting batch processing...")
        for start_idx in range(0, len(df), batch_size):
            batch = df.iloc[start_idx : start_idx + batch_size]
            bulk_index_recipe_batch(batch, es_client)
            print(
                f"Processed batch {start_idx//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}"
            )

    except Exception as e:
        print(f"Error indexing recipes data: {e}")


def initialize_elasticsearch():
    """Main function to initialize Elasticsearch and index data"""
    try:
        # Create Elasticsearch client
        print("Creating Elasticsearch client...")
        es = create_elasticsearch_client()

        # Test connection
        if es.ping():
            print("Successfully connected to Elasticsearch")
        else:
            print("Could not connect to Elasticsearch")
            return

        # Create indices
        print("\nCreating indices...")
        create_indices(es)

        # Check number of documents in recipes index
        doc_count = es.count(index="recipes")["count"]
        print(f"\nCurrent number of documents in recipes index: {doc_count}")

        if doc_count < 5000:
            # Index recipes data
            print("\nIndexing recipes data...")
            index_recipes_data(es)
        else:
            print(
                "\nSkipping indexing as recipes index already has sufficient documents."
            )

        print("\nInitialization complete!")

    except Exception as e:
        print(f"Error during initialization: {e}")


initialize_elasticsearch()
