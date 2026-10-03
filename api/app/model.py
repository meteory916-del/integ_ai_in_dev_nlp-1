import joblib
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.model = None
        self.classes = None
    
    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Модель не найдена: {path}")
        self.model = joblib.load(path)
        self.classes = list(self.model.classes_)
        logger.info(f"✅ Модель загружена из {path}")
    
    def is_loaded(self) -> bool:
        return self.model is not None
    
    def predict(self, text: str) -> Dict:
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена")
        prediction = self.model.predict([text])
        probabilities = self.model.predict_proba([text])
        return {"text": text, "label": prediction[0], "score": round(float(max(probabilities[0])), 4)}
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(text) for text in texts]

ml_service = MLService()
