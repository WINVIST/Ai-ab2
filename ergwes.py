import pandas as pd
import sqlite3
import socket
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score, classification_report, completeness_score, homogeneity_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Масштабирование данных для кластеризации
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Кодирование целевых меток
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Иерархическая кластеризация
hierarchical = AgglomerativeClustering(n_clusters=len(set(y_encoded)), linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# DBSCAN кластеризация
dbscan = DBSCAN(eps=1.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Метрики для иерархической кластеризации
completeness_hierarchical = completeness_score(y_encoded, hierarchical_labels)
homogeneity_hierarchical = homogeneity_score(y_encoded, hierarchical_labels)

# Метрики для DBSCAN
# Удаляем шум (-1) из кластеров для метрик
dbscan_labels_filtered = dbscan_labels[dbscan_labels != -1]
y_encoded_filtered = y_encoded[dbscan_labels != -1]
completeness_dbscan = completeness_score(y_encoded_filtered, dbscan_labels_filtered)
homogeneity_dbscan = homogeneity_score(y_encoded_filtered, dbscan_labels_filtered)

# Вывод результатов кластеризации
print("Иерархическая кластеризация:")
print(f"Полнота: {completeness_hierarchical:.2f}")
print(f"Однородность: {homogeneity_hierarchical:.2f}")

print("\nDBSCAN кластеризация:")
print(f"Полнота: {completeness_dbscan:.2f}")
print(f"Однородность: {homogeneity_dbscan:.2f}")
