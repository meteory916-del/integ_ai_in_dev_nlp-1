from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import torch

# Загрузка данных
df = pd.read_csv('data/processed/train.csv')

# Загрузка модели и токенизатора
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=8
)

# Токенизация
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Подготовка данных
from datasets import Dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Разделение на train/test
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# Обучение
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
)

trainer.train()

# Сохранение
model.save_pretrained("./bank_classifier")
tokenizer.save_pretrained("./bank_classifier")
print("✅ Transformer-модель сохранена")