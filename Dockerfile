# Dockerfile — Customer Support Resolution OpenEnv

# Build:   docker build -t customer-support-env .
# Run:     docker run -p 8000:8000 customer-support-env


FROM python:3.11-slim

# Hugging Face Spaces metadata
LABEL org.opencontainers.image.title="Customer Support Resolution Environment"
LABEL org.opencontainers.image.description="OpenEnv-compatible customer support ticket resolution environment"
LABEL hf_space_sdk="docker"
LABEL openenv="true"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY customer_support_env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY customer_support_env/ /app/customer_support_env/
COPY inference.py /app/inference.py

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "customer_support_env.server:app", "--host", "0.0.0.0", "--port", "8000"]
