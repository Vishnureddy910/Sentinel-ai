# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV (Updated for newer Debian versions)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copy your requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the Sentinel-AI project into the container
COPY . .