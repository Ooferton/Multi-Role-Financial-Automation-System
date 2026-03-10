# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    procps \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies in steps
RUN pip install --no-cache-dir numpy pandas scipy
RUN pip install --no-cache-dir streamlit requests yfinance
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories and ensure entrypoint is executable
RUN mkdir -p data logs ml/models && \
    sed -i 's/\r$//' hf_entrypoint.sh && \
    chmod +x hf_entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port 7860
EXPOSE 7860

# Set the entrypoint
CMD ["./hf_entrypoint.sh"]
