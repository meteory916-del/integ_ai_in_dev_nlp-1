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
        # LinearSVC не имеет classes_ напрямую, берём из pipeline или SVC
        if hasattr(self.model, 'classes_'):
            self.classes = list(self.model.classes_)
        elif hasattr(self.model, 'named_steps') and 'svc' in self.model.named_steps:
            self.classes = list(self.model.named_steps['svc'].classes_)
        elif hasattr(self.model, 'svc'):
            self.classes = list(self.model.svc.classes_)
        else:
            self.classes = None
        logger.info(f"✅ Модель загружена из {path}")
        logger.info(f"Классы: {self.classes}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> Dict:
        if not self.is_loaded():
            raise RuntimeError("Модель не загружена")
        prediction = self.model.predict([text])
        # LinearSVC не имеет predict_proba, используем decision_function
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba([text])
            score = round(float(max(probabilities[0])), 4)
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function([text])[0]
            score = round(float(max(scores)), 4)
        else:
            score = 1.0
        return {"text": text, "label": str(prediction[0]), "score": score}

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(text) for text in texts]


ml_service = MLService()
