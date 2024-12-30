# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy ONLY the requirements file first
COPY requirements.txt .
RUN pip install uv
# Install dependencies - this layer will be cached unless requirements.txt changes
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
# RUN python elasticSearchInitialisation.py
# Run FastAPI on uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
