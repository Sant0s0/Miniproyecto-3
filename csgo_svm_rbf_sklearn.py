import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def cargar_datos(filepath='csgo_round_snapshots.csv', train_size=0.20):
    '''
    Carga el dataset y realiza un split 20% entrenamiento / 80% prueba.
    Permite entrenar la SVM RBF con ~24,482 muestras (computacionalmente viable)
    y evaluar su generalización en las ~97,928 muestras restantes de prueba.
    '''
    df = pd.read_csv(filepath)
    # Mapeo binario: CT = +1, T = -1
    df['round_winner'] = df['round_winner'].map({'CT': 1, 'T': -1})
    df['bomb_planted'] = df['bomb_planted'].astype(str).str.lower().eq('true').astype(int)
    df = pd.get_dummies(df, columns=['map'], drop_first=True, dtype=int)

    X = df.drop(columns=['round_winner']).astype(float)
    y = df['round_winner'].values

    # 20% train (~24.5k) y 80% test (~98k) estratificado
    return train_test_split(X, y, train_size=train_size, test_size=1.0 - train_size, stratify=y, random_state=42)


def calibrar_y_entrenar_rbf(X_train, y_train, n_submuestra=2000):
    '''
    Entrena la SVM RBF en 2 etapas:
    1. Calibra C y gamma rápidamente en una submuestra representativa (por defecto 2,000).
    2. Re-entrena el mejor estimador sobre el 100% de X_train.
    '''
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", cache_size=1000))
    ])

    param_grid = {
        "svm__C": [0.1, 1.0, 10.0],
        "svm__gamma": ["scale", 0.01, 0.1]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="recall", n_jobs=-1)

    if n_submuestra is not None and len(X_train) > n_submuestra:
        print(f"Calibrando C y gamma en submuestra de {n_submuestra:,} registros...")
        X_busq, _, y_busq, _ = train_test_split(
            X_train, y_train, train_size=n_submuestra, stratify=y_train, random_state=42
        )
        t0_sub = time.perf_counter()
        grid.fit(X_busq, y_busq)
        print(f"-> Hiperparámetros óptimos encontrados en {time.perf_counter() - t0_sub:.2f} s: {grid.best_params_}")
        
        print(f"Re-entrenando modelo final con las {len(X_train):,} muestras de entrenamiento...")
        t0_refit = time.perf_counter()
        grid.best_estimator_.fit(X_train, y_train)
        print(f"-> Re-entrenamiento completado en {time.perf_counter() - t0_refit:.2f} s.")
    else:
        print("Calibrando hiperparámetros C y gamma...")
        grid.fit(X_train, y_train)

    return grid


def evaluar_modelo(modelo_grid, X_test, y_test):
    '''Evalúa el modelo y devuelve el reporte y los mejores hiperparámetros.'''
    y_pred = modelo_grid.predict(X_test)
    reporte = classification_report(y_test, y_pred, target_names=['Terrorista (-1)', 'Antiterrorista (+1)'])
    return modelo_grid.best_params_, reporte


if __name__ == "__main__":
    t_inicio_total = time.perf_counter()

    # Partición modulable: ajusta aquí la proporción que desees (ej. 0.80 para 80/20, 0.20 para 20/80)
    PROPORCION_TRAIN = 0.80
    
    print(f"Cargando dataset (split {int(PROPORCION_TRAIN*100)}% train / {int((1-PROPORCION_TRAIN)*100)}% test)...")
    t0_carga = time.perf_counter()
    X_train, X_test, y_train, y_test = cargar_datos(train_size=PROPORCION_TRAIN)
    t_carga = time.perf_counter() - t0_carga
    print(f"-> Datos cargados: {len(y_train):,} train, {len(y_test):,} test en {t_carga:.4f} s.")

    t0_entrenamiento = time.perf_counter()
    modelo_optimizado = calibrar_y_entrenar_rbf(X_train, y_train)
    t_entrenamiento = time.perf_counter() - t0_entrenamiento

    t0_evaluacion = time.perf_counter()
    mejores, reporte = evaluar_modelo(modelo_optimizado, X_test, y_test)
    t_evaluacion = time.perf_counter() - t0_evaluacion

    svm = modelo_optimizado.best_estimator_.named_steps["svm"]
    print(f"\n¡Calibración exitosa! Mejores hiperparámetros: C = {mejores['svm__C']}, gamma = {mejores['svm__gamma']}")
    print(f"Vectores de soporte: {svm.n_support_.sum()} de {len(y_train)} muestras de entrenamiento")
    print("\nReporte de Evaluación en Prueba (Test):\n")
    print(reporte)

    t_total = time.perf_counter() - t_inicio_total
    print("\n=== TIEMPOS DE CÓMPUTO ===")
    print(f"Tiempo de carga y preparación de datos:          {t_carga:.4f} s ({t_carga/60:.2f} min)")
    print(f"Tiempo de calibración y entrenamiento (GridCV): {t_entrenamiento:.4f} s ({t_entrenamiento/60:.2f} min)")
    print(f"Tiempo de evaluación en prueba:                  {t_evaluacion:.4f} s")
    print(f"Tiempo total transcurrido:                       {t_total:.4f} s ({t_total/60:.2f} min)")
