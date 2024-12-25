from fastapi import FastAPI, Request, HTTPException
from utils import *

# from globals import df, distinct_ingredients, cuisines, courses, diets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
import logging
from functools import wraps
import traceback

# Add these near the top of your file, after imports
logging.basicConfig(
    filename="api_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_error(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Get the request object from args
            request = next((arg for arg in args if isinstance(arg, Request)), None)

            # Get the endpoint name
            endpoint = func.__name__

            try:
                # Try to get the request data
                data = await request.json() if request else "No request data"
            except:
                data = "Could not parse request data"

            # Log the error with detailed information
            logging.error(
                f"\nEndpoint: {endpoint}"
                f"\nURL: {request.url if request else 'No URL'}"
                f"\nMethod: {request.method if request else 'No method'}"
                f"\nData: {data}"
                f"\nError: {str(e)}"
                f"\nTraceback: {traceback.format_exc()}"
            )

            # Re-raise the exception
            raise HTTPException(status_code=500, detail=str(e))

    return wrapper


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    global df, distinct_ingredients, cuisines, courses, diets
    distinct_ingredients, cuisines, courses, diets = initialize_globals()
    yield


app = FastAPI(title="Recipe Recommendation API", lifespan=lifespan)


@app.get("/dropdown-data/")
@log_error
async def get_dropdown_data():
    return {
        "cuisines": cuisines,
        "courses": courses,
        "diets": diets,
        "ingredients": distinct_ingredients,
    }


@app.get("/")
@log_error
async def health_check():
    return {"Api is up"}


@app.post("/add-recipe/")
@log_error
async def add_recipe_endpoint(request: Request):
    try:
        data = await request.json()
        recipe = {
            "Title": data.get("recipe_name"),
            "Prep_Time": data.get("prep_time"),
            "Cook_Time": data.get("cook_time"),
            "Cuisine": data.get("selected_cuisines", []),
            "Course": data.get("selected_courses", []),
            "Diet": data.get("selected_diets", []),
            "Cleaned_Ingredients": data.get("selected_ingredients", []),
            "Image": data.get("image_input"),
        }
        add_recipe(recipe)
        return {"status": "Recipe added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Recipe not added")


@app.post("/save-review/")
@log_error
async def save_review_endpoint(request: Request):
    try:
        data = await request.json()
        save_review(data.get("review_text"))
        return {"status": "Review saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Review not saved")


@app.post("/submit-feedback/")
@log_error
async def submit_feedback(request: Request):
    # try:
    # Read the JSON data from the request body
    data = await request.json()

    # Extract the relevant fields from the JSON data
    user_id = data.get("user_id")
    recipe_titles = data.get("recipe_titles", [])
    rating = data.get("rating")
    title_text = data.get("title_text")

    image = data.get("image")
    try:
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))
    except:
        image = None
    # Save feedback
    save_feedback(user_id, recipe_titles, rating, title_text, image)
    return {"message": "Feedback received"}


# except Exception as e:
#     raise HTTPException(status_code=500, detail="Feedback not received")


# Endpoint to handle predictions
@app.post("/predict/")
@log_error
async def predict(request: Request):
    try:
        data = await request.json()
        recipe_titles, details = predict_recipes(data, df)
        return {"titles": recipe_titles, "details": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
