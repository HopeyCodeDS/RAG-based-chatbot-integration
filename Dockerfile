FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data directory first (add this)
COPY data ./data/

# Copy the rest of the application code
COPY . .

# Create and set permissions for chroma directory
RUN mkdir -p /app/chroma && chmod 777 /app/chroma

# Pre-compile Python files
RUN python -m compileall .

EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Add logging to stdout for gunicorn
CMD python populate_database.py --reset && \
    gunicorn app.main:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 600 \
    --bind 0.0.0.0:8000 \
    --log-level debug \
    --preload \
    --access-logfile - \
    --error-logfile - \
    --capture-output