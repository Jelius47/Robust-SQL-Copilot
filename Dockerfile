# FROM python:3.12

# # Set the working directory
# WORKDIR /app

# RUN apt-get update && apt-get install -y unixodbc-dev

# # Copy requirements and install Python dependencies (including Playwright)
# COPY requirements.txt /app/

# RUN pip install -r requirements.txt

# # Copy the application and Jupyter configuration
# COPY . /app

# EXPOSE 8080

# # Start Jupyter Notebook and Python application
# CMD ["chainlit","run","assistant.py","--port","8080"]


FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies including build tools for chroma-hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    build-essential \
    gcc \
    g++ \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Only copy requirements.txt to allow caching of pip install
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Now copy the rest of the application
COPY . .

# Expose the port for Chainlit
EXPOSE 8080

# Default command
CMD ["chainlit", "run", "assistant.py", "--port", "8080" ,"--watch"]
