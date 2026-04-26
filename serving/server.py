"""ONNX export + FastAPI serving with Prometheus metrics.

Endpoints:
  GET  /health         — liveness check
  POST /predict        — direct feature input inference
  POST /predict/v2     — user_id/ad_id → Redis feature lookup → inference
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX Export (requires torch — imported lazily so serving works without it)
# ---------------------------------------------------------------------------

def export_onnx(model, config: dict, output_path: str = None):
    """Export PyTorch model to ONNX format with sigmoid applied."""
    import onnx
    import torch

    class _SigmoidWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, dense, sparse):
            return torch.sigmoid(self.inner(dense, sparse))

    output_path = output_path or config.get("serving", {}).get("onnx_path", "checkpoints/model.onnx")
    model.eval()
    model.cpu()

    wrapped = _SigmoidWrapper(model)
    wrapped.eval()

    mc = config.get("model", {})
    dc = config.get("data", {})
    num_dense = mc.get("num_dense", 13)
    num_sparse = mc.get("num_sparse_fields", 26)

    dummy_dense = torch.randn(1, num_dense)
    dummy_sparse = torch.randint(0, dc.get("hash_bucket_size", 100_000), (1, num_sparse))

    torch.onnx.export(
        wrapped, (dummy_dense, dummy_sparse), output_path,
        input_names=["dense", "sparse"],
        output_names=["prediction"],
        dynamic_axes={"dense": {0: "batch"}, "sparse": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info(f"ONNX model exported and verified: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

def create_app(config_path: str = "configs/default.yaml"):
    """Create FastAPI app — called by uvicorn or main.py."""
    import onnxruntime as ort
    from fastapi import FastAPI, HTTPException
    from prometheus_client import Counter, Histogram, generate_latest
    from pydantic import BaseModel
    from starlette.concurrency import run_in_threadpool
    from starlette.responses import Response

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        config = {}

    sc = config.get("serving", {})
    rc = config.get("redis", {})
    mc = config.get("model", {})
    max_batch = sc.get("max_batch_size", 256)

    # Env vars override config (for Docker)
    redis_host = os.environ.get("REDIS_HOST", rc.get("host", "localhost"))
    redis_port = int(os.environ.get("REDIS_PORT", rc.get("port", 6379)))

    # State
    state: Dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        onnx_path = sc.get("onnx_path", "checkpoints/model.onnx")
        state["session"] = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        logger.info(f"Loaded ONNX model: {onnx_path}")

        # Optional Redis
        try:
            import redis.asyncio as aioredis
            state["redis"] = aioredis.Redis(
                host=redis_host,
                port=redis_port,
                db=rc.get("db", 0),
                decode_responses=True,
            )
            await state["redis"].ping()
            logger.info("Redis connected")
        except Exception:
            state["redis"] = None
            logger.warning("Redis unavailable — /predict/v2 disabled")

        yield
        if state.get("redis"):
            await state["redis"].close()

    app = FastAPI(title="CTR Prediction API", lifespan=lifespan)

    # Prometheus metrics
    REQUEST_COUNT = Counter("ctr_requests_total", "Total requests", ["endpoint", "status"])
    REQUEST_LATENCY = Histogram(
        "ctr_request_latency_seconds", "Request latency", ["endpoint"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    PREDICTION_VALUE = Histogram(
        "ctr_prediction_value", "Predicted CTR distribution",
        buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
    )

    # Request schemas
    class PredictRequest(BaseModel):
        dense_features: List[float]
        sparse_features: List[int]

    class BatchPredictRequest(BaseModel):
        samples: List[PredictRequest]

    class PredictV2Request(BaseModel):
        user_id: str
        ad_id: str
        context: Optional[Dict] = None

    class PredictResponse(BaseModel):
        prediction: float
        latency_ms: float

    class BatchPredictResponse(BaseModel):
        predictions: List[float]
        latency_ms: float

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": "session" in state}

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type="text/plain")

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        num_dense = mc.get("num_dense", 13)
        num_sparse = mc.get("num_sparse_fields", 26)
        if len(req.dense_features) != num_dense:
            raise HTTPException(400, f"dense_features must have {num_dense} elements, got {len(req.dense_features)}")
        if len(req.sparse_features) != num_sparse:
            raise HTTPException(400, f"sparse_features must have {num_sparse} elements, got {len(req.sparse_features)}")
        t0 = time.time()
        dense = np.array([req.dense_features], dtype=np.float32)
        sparse = np.array([req.sparse_features], dtype=np.int64)
        try:
            result = await run_in_threadpool(state["session"].run, None, {"dense": dense, "sparse": sparse})
            pred = float(result[0][0])
            latency = (time.time() - t0) * 1000
            REQUEST_COUNT.labels(endpoint="/predict", status="ok").inc()
        except Exception:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise

        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency / 1000)
        PREDICTION_VALUE.observe(pred)
        return PredictResponse(prediction=pred, latency_ms=round(latency, 2))

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(req: BatchPredictRequest):
        if len(req.samples) > max_batch:
            raise HTTPException(400, f"Batch size exceeds limit ({max_batch})")
        t0 = time.time()
        dense = np.array([s.dense_features for s in req.samples], dtype=np.float32)
        sparse = np.array([s.sparse_features for s in req.samples], dtype=np.int64)
        try:
            result = await run_in_threadpool(state["session"].run, None, {"dense": dense, "sparse": sparse})
            preds = result[0].tolist()
            latency = (time.time() - t0) * 1000
            REQUEST_COUNT.labels(endpoint="/predict/batch", status="ok").inc()
        except Exception:
            REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
            raise

        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(latency / 1000)
        return BatchPredictResponse(predictions=preds, latency_ms=round(latency, 2))

    # Flink-produced feature keys in Redis (must match FeatureStoreSink output)
    ONLINE_FEATURE_KEYS = {
        "user": ["user_ctr_5m_ctr", "user_ctr_5m_impressions", "user_ctr_5m_clicks",
                 "user_ctr_1h_ctr", "user_ctr_1h_impressions", "user_ctr_1h_clicks",
                 "user_ctr_24h_ctr", "user_ctr_24h_impressions", "user_ctr_24h_clicks"],
        "ad":   ["ad_ctr_1h_ctr", "ad_ctr_1h_impressions",
                 "ad_ctr_24h_ctr", "ad_ctr_24h_impressions"],
    }

    def _assemble_features(user_vals: dict, ad_vals: dict, context: dict) -> tuple:
        """Assemble Flink-computed online features + context into model input vectors.

        Dense (13): 9 online features (user/ad CTR & counts) + 4 context features
        Sparse (26): hash-encoded context fields with per-field offset, zero-padded.

        Note: The Criteo training model uses 13 dense + 26 sparse features. In online serving,
        dense features come from Flink-computed aggregates (user/ad CTR stats) and sparse
        features from request context. Unused sparse slots are zero-padded; padding_idx=0
        ensures these contribute zero to the model output.
        """
        import hashlib as _hl
        hbs = config.get("data", {}).get("hash_bucket_size", 100_000)

        dense = [
            float(user_vals.get("user_ctr_5m_ctr", 0)),
            float(user_vals.get("user_ctr_5m_impressions", 0)),
            float(user_vals.get("user_ctr_5m_clicks", 0)),
            float(user_vals.get("user_ctr_1h_ctr", 0)),
            float(user_vals.get("user_ctr_1h_impressions", 0)),
            float(user_vals.get("user_ctr_1h_clicks", 0)),
            float(user_vals.get("user_ctr_24h_ctr", 0)),
            float(user_vals.get("user_ctr_24h_impressions", 0)),
            float(user_vals.get("user_ctr_24h_clicks", 0)),
            float(ad_vals.get("ad_ctr_1h_ctr", 0)),
            float(ad_vals.get("ad_ctr_1h_impressions", 0)),
            float(ad_vals.get("ad_ctr_24h_ctr", 0)),
            float(ad_vals.get("ad_ctr_24h_impressions", 0)),
        ]

        ctx = context or {}
        ctx_fields = [
            ctx.get("device", ""), ctx.get("os", ""), ctx.get("geo", ""),
            ctx.get("slot", ""), str(ctx.get("hour", 0)), str(ctx.get("day_of_week", 0)),
        ]
        sparse = [i * hbs + int(_hl.md5(v.encode()).hexdigest(), 16) % hbs for i, v in enumerate(ctx_fields)]
        # Zero-pad remaining sparse fields — padding_idx=0 maps to zero vector
        sparse += [0] * (26 - len(sparse))

        return (
            np.array([dense], dtype=np.float32),
            np.array([sparse], dtype=np.int64),
        )

    @app.post("/predict/v2", response_model=PredictResponse)
    async def predict_v2(req: PredictV2Request):
        """Feature Store lookup (Flink-computed features) → inference."""
        if not state.get("redis"):
            raise HTTPException(503, "Redis unavailable")
        t0 = time.time()

        r = state["redis"]

        # Fetch online features from Redis (written by Flink FeatureStoreSink)
        user_raw = await r.hgetall(f"user:{req.user_id}")
        ad_raw = await r.hgetall(f"ad:{req.ad_id}")

        dense, sparse = _assemble_features(user_raw, ad_raw, req.context)

        try:
            result = await run_in_threadpool(state["session"].run, None, {"dense": dense, "sparse": sparse})
            pred = float(result[0][0])
            latency = (time.time() - t0) * 1000
            REQUEST_COUNT.labels(endpoint="/predict/v2", status="ok").inc()
        except Exception:
            REQUEST_COUNT.labels(endpoint="/predict/v2", status="error").inc()
            raise

        REQUEST_LATENCY.labels(endpoint="/predict/v2").observe(latency / 1000)
        PREDICTION_VALUE.observe(pred)
        return PredictResponse(prediction=pred, latency_ms=round(latency, 2))

    return app


app = create_app(os.environ.get("CTR_CONFIG_PATH", "configs/default.yaml"))
