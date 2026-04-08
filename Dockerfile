# Dockerfile — Customer Support Resolution OpenEnv
#
# Build:  docker build -t customer-support-env .
# Run:    docker run -p 7860:7860 customer-support-env

FROM python:3.11-slim

# Hugging Face Spaces / OpenEnv metadata
LABEL org.opencontainers.image.title="Customer Support Resolution Environment"
LABEL org.opencontainers.image.description="OpenEnv-compatible customer support ticket resolution environment"
LABEL hf_space_sdk="docker"
LABEL openenv="true"

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────
# Copy requirements first for better layer caching
COPY customer_support_env/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────
# Root-level config files required by the OpenEnv validator
COPY pyproject.toml  /app/pyproject.toml
COPY openenv.yaml    /app/openenv.yaml
COPY README.md       /app/README.md

# Package source
COPY customer_support_env/ /app/customer_support_env/

# Inference script (evaluated by the validator)
COPY inference.py /app/inference.py

# Server wrapper (used by HF Spaces / OpenEnv runtime)
COPY server/ /app/server/

# ── Environment ───────────────────────────────────────────────────────────
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Expose port ───────────────────────────────────────────────────────────
EXPOSE 7860

# ── Start the FastAPI server ──────────────────────────────────────────────
CMD ["uvicorn", "customer_support_env.server:app", "--host", "0.0.0.0", "--port", "7860"]
