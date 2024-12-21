from fastapi import FastAPI, Request, HTTPException
from utils import *
from globals import model, df, distinct_ingredients, cuisines, courses, diets
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    global df, distinct_ingredients, cuisines, courses, diets
    df, distinct_ingredients, cuisines, courses, diets = initialize_globals()
    yield


app = FastAPI(title="Recipe Recommendation API", lifespan=lifespan)


@app.get("/dropdown-data/")
async def get_dropdown_data():
    return {
        "cuisines": cuisines,
        "courses": courses,
        "diets": diets,
        "ingredients": distinct_ingredients,
    }


@app.get("/")
async def health_check():
    return {"Api is up"}


@app.post("/add-recipe/")
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
async def save_review_endpoint(request: Request):
    try:
        data = await request.json()
        save_review(data.get("review_text"))
        return {"status": "Review saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Review not saved")


@app.post("/submit-feedback/")
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
async def predict(request: Request):
    try:
        data = await request.json()
        recipe_titles, details = predict_recipes(data, df)
        return {"titles": recipe_titles, "details": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Load the DataFrame
# df = pd.read_csv("./data.csv")
# df["Cleaned_Ingredients"] = df["Cleaned_Ingredients"].apply(ast.literal_eval)


# # Compute distinct ingredients, cuisines, courses, and diets
# distinct_ingredients = set()
# for row in df["Cleaned_Ingredients"]:
#     for ingredient in row:
#         distinct_ingredients.add(ingredient)
# distinct_ingredients = sorted(list(distinct_ingredients))

# cuisines = df["Cuisine"].dropna().unique().tolist()
# cuisines = [cuisine for cuisine in cuisines if cuisine.lower() != "unknown"]

# courses = df["Course"].dropna().unique().tolist()
# courses = [course for course in courses if course.lower() != "unknown"]

# diets = df["Diet"].dropna().unique().tolist()
# diets = [diet for diet in diets if diet.lower() != "unknown"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
