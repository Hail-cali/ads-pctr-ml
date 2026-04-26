FROM python:3.11-slim

WORKDIR /app

# Serving dependencies only (no torch)
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    "uvicorn[standard]>=0.23.0" \
    pydantic>=2.0.0 \
    onnxruntime>=1.15.0 \
    prometheus-client>=0.17.0 \
    pyyaml>=6.0 \
    numpy>=1.24.0 \
    "redis>=5.0.0"

COPY configs/ configs/
COPY serving/ serving/
COPY checkpoints/ checkpoints/

EXPOSE 8000

CMD ["uvicorn", "serving.server:app", "--host", "0.0.0.0", "--port", "8000"]
