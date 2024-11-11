# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for MLflow (optional)
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# Run training script by default (you can specify other entry points if needed)
CMD ["python", "train.py"]
