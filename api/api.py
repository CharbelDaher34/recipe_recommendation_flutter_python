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
    # try:
    data = await request.json()

    # Access data directly from the request body (as a dictionary)
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
        return "No matching recipes found. Please adjust your inputs.", filtered_df[
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
    if image is not None:
        # read the image
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))

    # Compute the average embedding
    avg_embedding = compute_average_embedding(title_text, image)
    # Load user embeddings
    user_embeddings = load_user_embeddings()
    user_embedding = get_user_embeddings(user_id, user_embeddings)
    avg_embedding = np.array(avg_embedding)
    user_embedding = np.array(user_embedding)
    if user_embedding is not None:
        avg_embedding = 0.8 * avg_embedding + 0.2 * user_embedding

    if avg_embedding is None:
        final_df = filtered_df.head(5)

    else:
        top_ids = find_most_similar_recipe(
            avg_embedding, "./embeddings.json", filtered_df, top_n=5
        )
        final_df = filtered_df[filtered_df["ID"].apply(lambda x: x in top_ids)]

    recipe_titles = final_df["Title"].tolist()
    details = (
        final_df[
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
        # .to_dict(orient="records")
    )

    return {"titles": recipe_titles, "details": details}


# except Exception as e:
#     raise HTTPException(status_code=500, detail=str(e))


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
