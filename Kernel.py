import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("csgo.csv")
df["bomb_planted"] = df["bomb_planted"].astype(int)
df = pd.get_dummies(df, columns=["map"], dtype=int)

X = df.drop(columns=["round_winner"])
y = df["round_winner"]


df_sample = df.sample(n=100000, random_state=42)

X = df_sample.drop(columns=["round_winner"])
y = df_sample["round_winner"]

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))