# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install uv
RUN pip install uv

# Copy the requirements first to leverage Docker cache
COPY requirements.txt .

# Install system dependencies and Python packages using uv
RUN uv pip install --system --no-cache-dir -r requirements.txt
# Pre-download the model directly
RUN python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)"

COPY saveJinaLocally.py .
RUN python saveJinaLocally.py
# Copy the application code
COPY . .
# Run the Python app directly
CMD ["python", "app.py"]
