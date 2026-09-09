# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
# and to ensure stdout is logged immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Java (JRE) required for PySpark, and procps for process management
RUN apt-get update && \
    apt-get install -y default-jre procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose ports for FastAPI (8000) and Streamlit (8501)
# Note: Render usually routes traffic to the first bound port (8000 or 8501 depending on setup).
# We'll configure Streamlit to be the main entrypoint port if exposed directly.
EXPOSE 8000
EXPOSE 8501

# Run the application using the start script
CMD ["./start.sh"]
