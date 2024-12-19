import pandas as pd
import sqlite3
import socket
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Проверка наличия файла базы данных
db_path = 'server_logs_600k.db'
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Database file not found at {db_path}")

# Подключение к базе данных SQLite
conn = sqlite3.connect(db_path)

# Загрузка данных из таблицы в DataFrame
data = pd.read_sql_query("SELECT * FROM logs", conn)

# Закрытие подключения к базе данных
conn.close()

# Преобразование временной метки в UNIX-время
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
data = data.dropna(subset=['Timestamp'])  # Удаление строк с некорректными временными метками
data['Timestamp'] = data['Timestamp'].astype('int64') // 10**9  # Преобразование в секунды

# Кодирование IP-адресов (например, преобразование в целые числа)
def encode_ip(ip):
    try:
        return int.from_bytes(socket.inet_aton(ip), 'big')
    except OSError:
        return None

data['IP'] = data['IP'].apply(encode_ip)
data = data.dropna(subset=['IP'])  # Удаление строк с некорректными IP-адресами

data['IP'] = data['IP'].astype('int64')

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['Attack Type'])  # Убедитесь, что вы удаляете только нужный столбец
y = data['Attack Type']

# Кодирование категориальных признаков
categorical_columns = ['HTTP Method', 'URL', 'User-Agent']
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Кодирование целевых меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Обучение модели
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Прогнозирование на обучающей выборке
y_pred_train = model.predict(X_train)

# Прогнозирование на тестовой выборке
y_pred_test = model.predict(X_test)

# Вычисление точности на обучающей выборке
accuracy_train = accuracy_score(y_train, y_pred_train) * 100  # Умножение на 100 для получения процентов
classification_rep_train = classification_report(y_train, y_pred_train, target_names=label_encoder.classes_)

# Вычисление точности на тестовой выборке
accuracy_test = accuracy_score(y_test, y_pred_test) * 100  # Умножение на 100 для получения процентов
classification_rep_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)

# Вывод результатов
print(f"Точность модели на обучающей выборке: {accuracy_train:.2f}%")
print("Отчет о классификации на обучающей выборке:\n", classification_rep_train)

print(f"Точность модели на тестовой выборке: {accuracy_test:.2f}%")
print("Отчет о классификации на тестовой выборке:\n", classification_rep_test)
