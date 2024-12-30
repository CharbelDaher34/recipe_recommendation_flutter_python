from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel
import torch
from PIL import Image
import base64
import io
import re
import itertools
from typing import Dict
from datetime import datetime


class InputString(BaseModel):
    content: str


class ModelInstance:
    def __init__(self, model_path: str):
        # Load the model using AutoModel instead of torch.load
        self.model = AutoModel.from_pretrained(
            "./jina_clip_v1_model", trust_remote_code=True
        )
        self.model = torch.load(f"{model_path}/jina.pt")
        self.model.eval()  # Set the model to evaluation mode
        self.request_count = 0
        print(f"Model loaded from {model_path}")

    def encode_input(self, input_string):
        """
        Encode either text or base64 image string using the CLIP model
        """
        self.request_count += 1
        try:
            if is_base64_image(input_string):
                # Process as image
                image = base64_to_image(input_string)
                embeddings = self.model.encode_image(image)
                return {
                    "status": "success",
                    "type": "image",
                    "shape": embeddings.shape,
                    "embeddings": embeddings.tolist(),
                }
            else:
                # Process as text
                embeddings = self.model.encode_text([input_string])
                return {
                    "status": "success",
                    "type": "text",
                    "shape": embeddings.shape,
                    "embeddings": embeddings.tolist(),
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}


class LoadBalancer:
    def __init__(self):
        self.instances: Dict[str, ModelInstance] = {}
        self.instance_cycle = None

    def add_instance(self, instance_id: str, model_path: str):
        self.instances[instance_id] = ModelInstance(model_path)
        self.instance_cycle = itertools.cycle(self.instances.keys())

    def get_next_instance(self) -> ModelInstance:
        if not self.instances:
            raise RuntimeError("No model instances available")
        next_id = next(self.instance_cycle)
        return self.instances[next_id]

    def get_stats(self):
        return {
            instance_id: {"requests_processed": instance.request_count}
            for instance_id, instance in self.instances.items()
        }


app = FastAPI()

load_balancer = LoadBalancer()


@app.post("/encode")
async def encode_string(input_data: InputString):
    instance = load_balancer.get_next_instance()
    result = instance.encode_input(input_data.content)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.get("/stats")
async def get_stats():
    return load_balancer.get_stats()


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}


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


if __name__ == "__main__":
    import uvicorn

    # Initialize model instances
    load_balancer.add_instance("model1", "./jina_clip_v1_model")
    load_balancer.add_instance("model2", "./jina_clip_v1_model")
    uvicorn.run(app, host="0.0.0.0", port=8000)
