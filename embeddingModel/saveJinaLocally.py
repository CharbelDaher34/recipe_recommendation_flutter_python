from transformers import AutoModel
import torch
import os
from tqdm import tqdm

model_path = "./jina_clip_v1_model"

# Check if the folder already exists
if not os.path.exists(model_path):
    print(f"Loading and saving model to {model_path}")
    os.makedirs(model_path, exist_ok=True)

    # Use tqdm to track download progress
    with tqdm(
        unit="B", unit_scale=True, miniters=1, desc="Downloading Model"
    ) as progress_bar:
        model = AutoModel.from_pretrained(
            "jinaai/jina-clip-v1",
            trust_remote_code=True,
        )

    model.eval()
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the device

    # Save the model
    model.save_pretrained(model_path, safe_serialization=False)
    torch.save(model, os.path.join(model_path, "jina.pt"))
    print("Model saved successfully")
else:
    print(f"Model folder {model_path} already exists. Skipping download.")
