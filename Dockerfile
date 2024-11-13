FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git

# Set work directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install dependencies without constraints
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the script directory to the container
COPY script/ /app/script

# Set default command
CMD ["python", "/app/script/train.py"]
