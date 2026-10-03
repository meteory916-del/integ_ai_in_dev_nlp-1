import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Загрузка данных
df = pd.read_csv('data/processed/train.csv')

# Проверьте названия колонок
print("Колонки:", df.columns.tolist())
print("Размер данных:", df.shape)
print("Классы:", df['label'].unique())

# Разделение на текст и метки
X = df['text'].fillna('')
y = df['label']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Создание пайплайна: TF-IDF + LogisticRegression
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )),
    ('clf', LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced'
    ))
])

# Обучение
print("\nОбучение модели...")
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Сохранение модели
joblib.dump(model, 'model.pkl')
print("\nМодель сохранена в model.pkl")
