from transformers import AutoModel
import torch
from PIL import Image
import base64
import io


# Test image encoding/decoding
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


model = AutoModel.from_pretrained("./jina_clip_v1_model", trust_remote_code=True)
model = torch.load("./jina_clip_v1_model/jina.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and process image
image = Image.open("../../multimediaProject-main/imageResults/image6.jpeg")
# Convert to base64 string
base64_string = image_to_base64(image)
# Convert back to image
decoded_image = base64_to_image(base64_string)

# Encode the decoded image
image_embeddings = model.encode_image(decoded_image)
print("Image embeddings shape:", image_embeddings.shape)

# Test text encoding for comparison
text_embeddings = model.encode_text(["hello", "world"])
print("Text embeddings shape:", text_embeddings.shape)
