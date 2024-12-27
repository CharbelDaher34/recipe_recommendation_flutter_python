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

# Copy the application code
COPY . .

# Run the script to save Jina model locally first, then start the FastAPI app
CMD python saveJinaLocally.py && python app.py 