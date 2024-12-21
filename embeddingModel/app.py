from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel
import torch
from PIL import Image
import base64
import io
import re

app = FastAPI()


# Helper functions
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


# Load model (do this once at startup)
model = AutoModel.from_pretrained("./jina_clip_v1_model", trust_remote_code=True)
model = torch.load("./jina_clip_v1_model/jina.pt", map_location=torch.device("cpu"))


class InputString(BaseModel):
    content: str


def encode_input(input_string):
    """
    Encode either text or base64 image string using the CLIP model
    """
    try:
        if is_base64_image(input_string):
            # Process as image
            image = base64_to_image(input_string)
            embeddings = model.encode_image(image)
            return {
                "status": "success",
                "type": "image",
                "shape": embeddings.shape,
                "embeddings": embeddings.tolist(),
            }
        else:
            # Process as text
            embeddings = model.encode_text([input_string])
            return {
                "status": "success",
                "type": "text",
                "shape": embeddings.shape,
                "embeddings": embeddings.tolist(),
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/encode")
async def encode_string(input_data: InputString):
    result = encode_input(input_data.content)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# Test endpoint
@app.get("/")
async def root():
    return {"message": "CLIP Encoding API is running"}


if __name__ == "__main__":
    import uvicorn

    # Test the API with some examples
    print("Testing text encoding...")
    test_text = encode_input("hello world")
    print(f"Text result: {test_text['type']}, Shape: {test_text['shape']}")

    print("\nTesting image encoding...")
    test_image = Image.open("../../compression_comparison.png")
    base64_string = image_to_base64(test_image)
    image_result = encode_input(base64_string)
    print(f"Image result: {image_result['type']}, Shape: {image_result['shape']}")

    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
