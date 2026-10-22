import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("csgo.csv")

df["bomb_planted"] = df["bomb_planted"].astype(int)
df = pd.get_dummies(df, columns=["map"], dtype=int)


df_sample = df.sample(n=20000, random_state=42)

X = df_sample.drop(columns=["round_winner"])
y = df_sample["round_winner"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Entrenamiento final:", len(X_train))
print("Prueba:", len(X_test))


X_busqueda, _, y_busqueda, _ = train_test_split(
    X_train,
    y_train,
    train_size=2000,
    random_state=42,
    stratify=y_train
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

parametros = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1]
}

busqueda = GridSearchCV(
    pipeline,
    param_grid=parametros,
    scoring="accuracy",
    cv=3,
    verbose=2
)

busqueda.fit(X_busqueda, y_busqueda)

print("Mejor C:", busqueda.best_params_["svm__C"])
print("Mejor gamma:", busqueda.best_params_["svm__gamma"])
print("Accuracy de validación:", busqueda.best_score_)

modelo = busqueda.best_estimator_

modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("Accuracy final:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
