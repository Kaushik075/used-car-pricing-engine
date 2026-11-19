# Prediction API image: Python + the model libraries + src/ + the saved model. No data, no training tools.
#   docker build -t used-car-pricing-api .
#   docker run --rm -p 8000:8000 used-car-pricing-api      -> http://localhost:8000/docs
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime (libgomp), which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is reused as long as requirements-api.txt does not change.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/ src/
COPY models/pricing_model.joblib models/metadata.json models/

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 api
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
