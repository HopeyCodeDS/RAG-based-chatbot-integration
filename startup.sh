#!/bin/bash
set -x  # Enable debug mode
cd /app
echo "Starting gunicorn..."
gunicorn app.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 600 \
    --bind 0.0.0.0:8000 \
    --log-level debug \
    --preload