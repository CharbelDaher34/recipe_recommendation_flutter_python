from fastapi import FastAPI, Request, HTTPException
from utils import (
    initialize_globals,
    add_recipe,
    save_review,
    index_feedback,
    predict_recipes,
    create_user,
    login_user,
)
from models import Recipe, User, Feedback, Review, RecipeAdd, UserReview
from datetime import timedelta, datetime

# from globals import df, distinct_ingredients, cuisines, courses, diets
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
import logging
from functools import wraps
import traceback
from auth_utils import require_auth, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from elasticSearchInitialisation import initialize_elasticsearch

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
    global df, distinct_ingredients, cuisines, courses, diets, es

    initialize_elasticsearch()

    # Initialize other globals
    distinct_ingredients, cuisines, courses, diets = initialize_globals()

    yield

    # Cleanup (if needed)
    if es is not None:
        es.close()


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
@require_auth
async def add_recipe_endpoint(recipe: Recipe, request: Request = None):
    """
    Add a new recipe to the pending recipes index

    Args:
        request: FastAPI Request object containing user authentication info
        recipe: Recipe model instance containing all recipe details
    """
    try:
        add_recipe(recipe)
        return {"status": "Recipe submitted for review successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add recipe: {str(e)}")


@app.post("/save-review/")
@log_error
@require_auth
async def save_review_endpoint(email: str, review: Review, request: Request):
    """
    Save a user review

    Args:
        email: User's email address
        review: Review model instance containing review content and timestamp
        request: FastAPI Request object containing user authentication info
    """
    try:
        success = save_review(email, review)
        if success:
            return {"status": "Review saved successfully"}
        raise HTTPException(status_code=500, detail="Failed to save review")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save review: {str(e)}")


@app.post("/submit-feedback/")
@log_error
@require_auth
async def submit_feedback(feedback: Feedback, request: Request):
    """
    Submit feedback endpoint that indexes feedback into Elasticsearch

    Args:
        feedback: Feedback model instance containing feedback details
        request: FastAPI Request object containing user authentication info

    Returns:
        dict: Contains success/failure message
    """
    try:
        success = index_feedback(feedback)
        if success:
            return {"message": "Feedback received successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to index feedback")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error submitting feedback: {str(e)}"
        )


# Endpoint to handle predictions
@app.post("/predict/")
@log_error
async def predict(request: Request):
    try:
        data = await request.json()
        details = predict_recipes(data)
        return {"details": details.to_json()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signup/")
@log_error
async def signup(user_input: User):
    try:
        # Note: In production, you should hash the password before storing
        # Try to create the user directly with user_input
        success = create_user(user_input)

        if success:
            return {"message": "User created successfully"}
        else:
            raise HTTPException(status_code=400, detail="User already exists")

    except ValueError as e:
        # This will catch Pydantic validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login/")
@log_error
async def login(user: User):
    """
    Login endpoint that verifies user credentials and returns an access token

    Args:
        user: User model instance containing email and password

    Returns:
        dict: Contains access token and token type on success
    """
    try:
        success = login_user(user)

        if success:
            # Create access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.email},  # "sub" is standard JWT claim for subject
                expires_delta=access_token_expires,
            )

            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")

    except ValueError as e:
        # This will catch Pydantic validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "api",
    }


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8001)
