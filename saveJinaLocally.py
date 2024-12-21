from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("jinaai/jina-clip-v1", trust_remote_code=True)
model.eval()
# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the device

model_path = "./api/jina_clip_v1_model"
model.save_pretrained(model_path, safe_serialization=False)
torch.save(model, "./api/jina_clip_v1_model/jina.pt")
