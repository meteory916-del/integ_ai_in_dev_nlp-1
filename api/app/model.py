import joblib
import os
import logging
import numpy as np
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
        if hasattr(self.model, 'classes_'):
            self.classes = [str(c).strip() for c in self.model.classes_]
        logger.info(f"✅ Модель загружена из {path}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> Dict:
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена")
        prediction = self.model.predict([text])
        label = str(prediction[0]).strip()
        
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function([text])
            max_score = float(np.max(scores[0]) if hasattr(scores, 'ndim') and scores.ndim > 1 else np.max(scores))
            # Сигмоида преобразует отрицательные значения decision_function в валидную вероятность [0, 1]
            score = float(1 / (1 + np.exp(-max_score)))
        elif hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba([text])
            score = float(np.max(probabilities[0]))
        else:
            score = 1.0
            
        return {"text": text, "label": label, "score": round(score, 4)}

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(text) for text in texts]

ml_service = MLService()
