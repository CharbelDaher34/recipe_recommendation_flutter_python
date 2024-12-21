import requests


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


# Test with a mix of text and base64-encoded images
import base64
from PIL import Image
import io


def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Convert an image to base64 first
image_base64 = image_to_base64("../../compression_comparison.png")
mixed_inputs = ["hello world", image_base64, "another text"]
mixed_results = get_embeddings(mixed_inputs)

# Print results
for result in mixed_results:
    if result["type"] == "error":
        print(f"Error for input '{result['input']}': {result['error']}")
    else:
        print(f"Successfully processed as {result['type']}")
        print(f"Embedding shape: {len(result['embeddings'])} dimensions")
