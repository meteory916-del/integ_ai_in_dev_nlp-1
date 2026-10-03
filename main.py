from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Банковская классификация NLP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели при старте
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Модель не найдена: {MODEL_PATH}. Запустите train_model.py")

model = joblib.load(MODEL_PATH)
print(f"✅ Модель загружена из {MODEL_PATH}")

# Допустимые классы
ALLOWED_LABELS = [
    "APP_LOGIN",
    "APP_TECH", 
    "CARD_ISSUE",
    "PAYMENT_OUT_FAIL",
    "INCOMING_DELAY",
    "FRAUD_SUSPECTED",
    "ACCOUNT_SEIZED",
    "KYC_VERIFICATION"
]

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    text: str
    category: str
    confidence: float

@app.get("/")
def root():
    return {"message": "API работает"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    try:
        # Предсказание
        prediction = model.predict([request.text])
        probabilities = model.predict_proba([request.text])
        
        category = prediction[0]
        confidence = float(max(probabilities[0]))
        
        # Проверка, что категория в списке допустимых
        if category not in ALLOWED_LABELS:
            raise HTTPException(
                status_code=500, 
                detail=f"Неизвестная категория: {category}"
            )
        
        return ClassifyResponse(
            text=request.text,
            category=category,
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   @app.get("/metrics")