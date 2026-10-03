from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging
import os
import time

from .schemas import ClassifyRequest, ClassifyResponse, HealthResponse, LabelEnum
from .model import ml_service

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === ПРОМЕТЕЙ МЕТРИКИ ===
REQUEST_COUNT = Counter(
    'classify_requests_total',
    'Total number of classification requests',
    ['label', 'status']
)

REQUEST_LATENCY = Histogram(
    'classify_latency_seconds',
    'Time spent processing classification request'
)

ERROR_COUNT = Counter(
    'classify_errors_total',
    'Total number of classification errors'
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether the model is loaded'
)

ALLOWED_LABELS = [e.value for e in LabelEnum]

# === ЗАГРУЗКА МОДЕЛИ ПРИ СТАРТЕ ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    model_path = os.path.normpath(model_path)
    logger.info(f"Путь к модели: {model_path}")
    
    try:
        ml_service.load_model(model_path)
        MODEL_LOADED.set(1)
        logger.info("✅ Модель успешно загружена!")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        MODEL_LOADED.set(0)
    
    yield
    logger.info("Приложение завершает работу")

# === ПРИЛОЖЕНИЕ ===
app = FastAPI(
    title="Банковская классификация NLP",
    version="0.1.0",
    description="API для классификации обращений клиентов банка",
    lifespan=lifespan
)

# === ЭНДПОИНТЫ ===
@app.get("/")
async def root():
    return {"message": "NLP API работает!", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model_loaded": ml_service.is_loaded(),
        "model_name": "TF-IDF + LogisticRegression"
    }

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    start = time.time()
    
    try:
        if not ml_service.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Модель не загружена"
            )
        
        result = ml_service.predict(request.text)
        label = result["label"]
        score = result["score"]
        
        if label not in ALLOWED_LABELS:
            ERROR_COUNT.inc()
            REQUEST_COUNT.labels(label=label, status="error").inc()
            raise HTTPException(
                status_code=500,
                detail=f"Неизвестная категория: {label}"
            )
        
        REQUEST_COUNT.labels(label=label, status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start)
        
        return ClassifyResponse(
            label=label,
            score=score
        )
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(label="unknown", status="error").inc()
        logger.error(f"Ошибка классификации: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Эндпоинт для Prometheus"""
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
