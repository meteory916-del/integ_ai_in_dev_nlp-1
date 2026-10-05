import pytest
import os
import sys

# Добавляем api/ в Python path
api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

# Загружаем модель ДО импорта app
from app.model import ml_service

model_path = os.path.join(api_dir, 'bank_classifier.joblib')
print(f"\nCONFTES: Загрузка модели из: {model_path}")

if os.path.exists(model_path) and not ml_service.is_loaded():
    ml_service.load_model(model_path)
    print("CONFTES: Модель загружена!")
elif ml_service.is_loaded():
    print("CONFTES: Модель уже загружена")
else:
    print(f"CONFTES: ОШИБКА - файл не найден: {model_path}")
