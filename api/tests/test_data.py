import pytest
import pandas as pd
import os

# Пути для поиска датасета (относительно папки api/tests/)
DATASET_PATHS = [
    "../../data/processed/train.csv",
    "../data/processed/train.csv",
    "data/processed/train.csv"
]

@pytest.fixture
def dataset_path():
    """Находит путь к train.csv."""
    for path in DATASET_PATHS:
        if os.path.exists(path):
            return path
    pytest.fail(f"Файл train.csv не найден по путям: {DATASET_PATHS}")


def test_dataset_exists_and_readable(dataset_path):
    """Тест: файл существует и читается pandas."""
    df = pd.read_csv(dataset_path)
    assert not df.empty, "Датасет пуст"


def test_dataset_has_required_columns(dataset_path):
    """Тест: в датасете есть колонки 'text' и 'label'."""
    df = pd.read_csv(dataset_path)
    assert "text" in df.columns, "Отсутствует колонка 'text'"
    assert "label" in df.columns, "Отсутствует колонка 'label'"


def test_dataset_labels_are_valid(dataset_path):
    """Тест: все метки в датасете принадлежат к 8 допустимым классам."""
    df = pd.read_csv(dataset_path)
    
    valid_labels = {
        "APP_LOGIN", "APP_TECH", "CARD_ISSUE", "PAYMENT_OUT_FAIL",
        "INCOMING_DELAY", "FRAUD_SUSPECTED", "ACCOUNT_SEIZED", "KYC_VERIFICATION"
    }
    
    actual_labels = set(df["label"].dropna().unique())
    invalid_labels = actual_labels - valid_labels
    
    assert len(invalid_labels) == 0, f"В датасете найдены недопустимые метки: {invalid_labels}"
