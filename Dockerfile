# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Configure Git to trust the /app directory
RUN git config --global --add safe.directory /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies with increased timeout and retries
RUN pip install --upgrade pip
RUN pip install --timeout=1200 --retries=5 -r requirements.txt

# Copy the rest of the application code
COPY script/ /app/script

# Expose Prometheus metrics port
EXPOSE 8000

# Set the default command to run your training script
CMD ["python", "/app/script/train.py"]
