# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Java (JRE) required for PySpark, and procps for process management
RUN apt-get update && \
    apt-get install -y default-jre procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Create a non-root user with UID 1000 (Required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Grant permissions to the non-root user for the app directory
RUN chown -R user:user /app
RUN chmod +x start.sh

# Switch to the non-root user
USER user

# Expose ports for FastAPI (8000) and Streamlit (7860 - default for HF Spaces)
EXPOSE 8000
EXPOSE 7860

# Run the application using the start script
CMD ["./start.sh"]
