import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler    
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def cargar_datos(filepath='csgo_round_snapshots.csv.xlsx'):
    '''Carga el dataset y lo prepara para clasificación binaria lineal.'''
    df = pd.read_excel(filepath)
    # Mapeo binario: CT = +1, T = -1
    df['round_winner'] = df['round_winner'].map({'CT': 1, 'T': -1})
    df = pd.get_dummies(df, columns=['map'], drop_first=True)
    
    X = df.drop(columns=['round_winner'])
    y = df['round_winner'].values
    
    # Split 80/20 clásico
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

def calibrar_y_entrenar_lineal(X_train, y_train):
    '''Entrena el modelo lineal y calibra el parámetro C usando GridSearchCV.'''
    # Regla de Oro 1 y 2: Estandarizar siempre y meterlo en el Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(dual=False, random_state=42, max_iter=5000))
    ])
    
    # Regla de Oro 3: AQUÍ SE CALIBRA EL MARGEN BLANDO (C)
    # Se prueba en escala logarítmica
    param_grid = {
        "svm__C": [0.001, 0.01, 0.1, 1.0, 10.0]
    }
    
    # Validacion cruzada de 5 particiones (cv=5) enfocada en Recall (Sensibilidad)
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="recall", n_jobs=-1)
    
    print("Calibrando hiperparámetro C... Esto puede tomar un minuto.")
    grid.fit(X_train, y_train)
    
    return grid

def evaluar_modelo(modelo_grid, X_test, y_test):
    '''Evalúa el modelo y devuelve el reporte y el mejor C.'''
    y_pred = modelo_grid.predict(X_test)
    reporte = classification_report(y_test, y_pred, target_names=['Terrorista (-1)', 'Antiterrorista (+1)'])
    mejor_c = modelo_grid.best_params_['svm__C']
    return mejor_c, reporte

if __name__ == "__main__":
    t_inicio_total = time.perf_counter()

    print("Cargando dataset...")
    t0_carga = time.perf_counter()
    X_train, X_test, y_train, y_test = cargar_datos()
    t_carga = time.perf_counter() - t0_carga
    print(f"-> Datos cargados en {t_carga:.4f} s.")

    t0_entrenamiento = time.perf_counter()
    modelo_optimizado = calibrar_y_entrenar_lineal(X_train, y_train)
    t_entrenamiento = time.perf_counter() - t0_entrenamiento

    t0_evaluacion = time.perf_counter()
    mejor_c, reporte = evaluar_modelo(modelo_optimizado, X_test, y_test)
    t_evaluacion = time.perf_counter() - t0_evaluacion
    
    print(f"\n¡Calibración exitosa! El mejor margen blando se logró con C = {mejor_c}")
    print("\nReporte de Evaluación en Prueba (Test):\n")
    print(reporte)

    t_total = time.perf_counter() - t_inicio_total
    print("\n=== TIEMPOS DE CÓMPUTO ===")
    print(f"Tiempo de carga y preparación de datos:          {t_carga:.4f} s ({t_carga/60:.2f} min)")
    print(f"Tiempo de calibración y entrenamiento (GridCV): {t_entrenamiento:.4f} s ({t_entrenamiento/60:.2f} min)")
    print(f"Tiempo de evaluación en prueba:                  {t_evaluacion:.4f} s")
    print(f"Tiempo total transcurrido:                       {t_total:.4f} s ({t_total/60:.2f} min)")