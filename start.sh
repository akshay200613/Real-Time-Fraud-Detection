#!/bin/bash
# Exit early on errors
set -e

# Start the FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for the backend to start
sleep 5

# Start the Streamlit frontend in the foreground
echo "Starting Streamlit frontend..."
streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0

# If Streamlit exits, kill the backend as well
kill $BACKEND_PID
