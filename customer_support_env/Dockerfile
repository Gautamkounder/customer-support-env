# Dockerfile — Customer Support Resolution OpenEnv

# Build:   docker build -t customer-support-env .
# Run:     docker run -p 8000:8000 customer-support-env


FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire environment package
COPY . /app/customer_support_env/

# Fix the package to be importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "customer_support_env.server:app", "--host", "0.0.0.0", "--port", "8000"]
