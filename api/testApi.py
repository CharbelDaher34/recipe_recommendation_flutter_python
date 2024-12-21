import requests
import base64
from PIL import Image
import io
import json

# API base URL
BASE_URL = "http://localhost:8001"


def print_response(name, response):
    print(f"\n=== {name} ===")
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {response.json()}")
    except requests.exceptions.JSONDecodeError:
        print(f"Raw Response: {response.text}")


def test_submit_feedback():
    try:
        # Create a sample image
        img = Image.new("RGB", (100, 100), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

        feedback_data = {
            "user_id": "test_user_123",
            "recipe_titles": ["Test Recipe 1", "Test Recipe 2"],
            "rating": 4,
            "title_text": "Test Recipe Feedback",
            "image": img_base64,
        }

        response = requests.post(f"{BASE_URL}/submit-feedback/", json=feedback_data)
        print_response("Submit Feedback", response)
    except Exception as e:
        print(f"\n=== Submit Feedback Error ===")
        print(f"Error: {str(e)}")


def test_predict():
    try:
        # Create a sample image
        img = Image.new("RGB", (100, 100), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")

        predict_data = {
            "user_id": "test_user_123",
            "title_text": "Spicy Chicken Curry",
            "prep_time": 30,
            "cook_time": 45,
            "selected_cuisines": ["Indian"],
            "selected_courses": ["Main Course"],
            "selected_diets": ["Non Vegetarian"],
            "selected_ingredients": ["Chicken", "Tomato"],
            "image": img_base64,
        }

        response = requests.post(f"{BASE_URL}/predict/", json=predict_data)
        print("\n=== Predict ===")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("Recipe Titles:", data.get("titles", []))
            print("\nFirst 200 characters of details:")
            print(data.get("details", "")[:200] + "...")
        else:
            print(f"Error Response: {response.text}")

    except Exception as e:
        print(f"\n=== Predict Error ===")
        print(f"Error: {str(e)}")


def test_health_check():
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response("Health Check", response)
    except Exception as e:
        print(f"\n=== Health Check Error ===")
        print(f"Error: {str(e)}")


def test_dropdown_data():
    try:
        response = requests.get(f"{BASE_URL}/dropdown-data/")
        print("\n=== Dropdown Data ===")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Sample data from each category:")
            for key in data:
                print(f"{key}: {data[key][:3]}...")  # Show first 3 items of each list
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"\n=== Dropdown Data Error ===")
        print(f"Error: {str(e)}")


def test_add_recipe():
    try:
        recipe_data = {
            "recipe_name": "Test Recipe",
            "prep_time": 30,
            "cook_time": 45,
            "selected_cuisines": ["Indian"],
            "selected_courses": ["Main Course"],
            "selected_diets": ["Vegetarian"],
            "selected_ingredients": ["Tomato", "Onion"],
            "image_input": None,
        }

        response = requests.post(f"{BASE_URL}/add-recipe/", json=recipe_data)
        print_response("Add Recipe", response)
    except Exception as e:
        print(f"\n=== Add Recipe Error ===")
        print(f"Error: {str(e)}")


def test_save_review():
    try:
        review_data = {"review_text": "This is a test review for the API"}

        response = requests.post(f"{BASE_URL}/save-review/", json=review_data)
        print_response("Save Review", response)
    except Exception as e:
        print(f"\n=== Save Review Error ===")
        print(f"Error: {str(e)}")


def main():
    print("Starting API Tests...")

    # Run all tests
    test_health_check()
    test_dropdown_data()
    test_add_recipe()
    test_save_review()
    test_submit_feedback()
    test_predict()


if __name__ == "__main__":
    main()
