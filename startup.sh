#!/bin/bash
cd /app
gunicorn app.main:app --timeout 600 --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000