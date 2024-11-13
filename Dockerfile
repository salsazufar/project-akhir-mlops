# Dockerfile
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt constraints.txt ./

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -c constraints.txt
