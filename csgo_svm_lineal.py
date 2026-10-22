import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def cargar_datos_manual(filepath='csgo_round_snapshots.csv.xlsx', n_samples=500):
    """
    Carga una muestra reducida del dataset para hacer viable la optimización 
    cuadrática manual (O(n^3)). Extrae 2 características para poder graficar.
    """
    df = pd.read_excel(filepath)
    
    # Mapeo de clases: CT = +1, T = -1
    df['round_winner'] = df['round_winner'].map({'CT': 1.0, 'T': -1.0})
    
    # Tomamos una muestra aleatoria balanceada (CT y T) preservando todas las columnas
    df_ct = df[df['round_winner'] == 1.0].sample(n_samples // 2, random_state=42)
    df_t = df[df['round_winner'] == -1.0].sample(n_samples // 2, random_state=42)
    df_sample = pd.concat([df_ct, df_t]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Seleccionamos 2 dimensiones fuertemente correlacionadas con la victoria: 
    # x1: Blindaje CT (ct_armor)
    # x2: Blindaje T (t_armor)
    X_raw = df_sample[['ct_armor', 't_armor']].values
    y = df_sample['round_winner'].values
    
    # Estandarización manual (Regla de Oro 1) para que el optimizador converja
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X = (X_raw - X_mean) / X_std
    
    return X, y

def entrenar_svm_lineal_manual(X_train, y_train, C=1.0):
    """
    Entrena la SVM lineal resolviendo el problema Dual de Lagrange paso a paso.
    El parámetro C (Margen blando) se recibe como argumento para que la futura GUI lo controle.
    """
    N = len(y_train)
    
    # 1. Matriz de Gram (K) e Interacción (P)
    K = np.dot(X_train, X_train.T)
    P = np.outer(y_train, y_train) * K
    
    # 2. Función de pérdida dual (para minimizar) y gradiente
    def dual_loss(alpha):
        return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)
    
    def grad_loss(alpha):
        return np.dot(P, alpha) - np.ones(N)
    
    # 3. Restricción de igualdad: sum(alpha * y) = 0
    restriccion_suma_cero = {
        'type': 'eq',
        'fun': lambda alpha: np.dot(alpha, y_train),
        'jac': lambda alpha: y_train
    }
    
    # 4. Límites de Margen Blando (0 <= alpha_i <= C)
    limites = [(0.0, float(C)) for _ in range(N)]
    alpha_inicial = np.zeros(N)
    
    # 5. Optimización Cuadrática
    resultado = minimize(
        dual_loss,
        alpha_inicial,
        jac=grad_loss,
        constraints=restriccion_suma_cero,
        bounds=limites,
        method='SLSQP'
    )
    
    alpha_opt = resultado.x
    alpha_opt[alpha_opt < 1e-5] = 0.0 # Limpieza de errores numéricos
    
    # 6. Recuperación de la geometría (w y b)
    w = np.sum((alpha_opt * y_train)[:, None] * X_train, axis=0)
    
    # Vectores libres (aquellos estrictamente dentro de la caja 0 < alpha < C)
    libres = np.where((alpha_opt > 1e-4) & (alpha_opt < (C - 1e-4)))[0]
    
    if len(libres) > 0:
        b = np.mean([y_train[s] - np.dot(w, X_train[s]) for s in libres])
    else:
        # Si no hay libres, promediamos sobre todos los vectores de soporte
        sv_todos = np.where(alpha_opt > 1e-4)[0]
        b = np.mean([y_train[s] - np.dot(w, X_train[s]) for s in sv_todos]) if len(sv_todos) > 0 else 0.0
        
    return {
        'alpha': alpha_opt,
        'w': w,
        'b': b,
        'C': C,
        'X_train': X_train,
        'y_train': y_train
    }

def predecir_manual(X_test, modelo):
    """
    Regla de decisión: y_hat = sign(w^T x + b)
    """
    w = modelo['w']
    b = modelo['b']
    f_x = np.dot(X_test, w) + b
    return np.sign(f_x)

if __name__ == "__main__":
    t_inicio_total = time.perf_counter()

    # Prueba del motor independiente
    n_muestras = 500  # ADVERTENCIA: 90000 muestras en optimización cuadrática manual O(N^3)
                       # generaría una matriz de 90000x90000 (~60 GB RAM). Ajustar si es necesario (ej. 1000).
    if n_muestras > 3000:
        mem_mb = (n_muestras ** 2 * 8) / (1024 ** 2)
        print(f"[!] ADVERTENCIA: n_samples={n_muestras} requerirá aprox. {mem_mb:.1f} MB solo para la matriz Gram")
        print("    y la optimización SLSQP O(N^3) puede saturar la CPU/RAM.")
        print("    Si se congela o se agota la memoria, cambia n_muestras a un valor menor (ej. 1000 o 2000).\n")

    print(f"Cargando muestra de datos ({n_muestras} muestras)...")
    t0_carga = time.perf_counter()
    X, y = cargar_datos_manual(n_samples=n_muestras)
    t_carga = time.perf_counter() - t0_carga
    print(f"-> Datos cargados en {t_carga:.4f} segundos.")
    
    # Calibración del Margen Blando: Aquí es donde tu futura GUI pasará el valor de C
    C_elegido = 1.0  
    print(f"\nOptimizando hiperplano con C = {C_elegido} (Margen Blando)...")
    
    t0_entrenamiento = time.perf_counter()
    modelo = entrenar_svm_lineal_manual(X, y, C=C_elegido)
    t_entrenamiento = time.perf_counter() - t0_entrenamiento
    
    w = modelo['w']
    b = modelo['b']
    ancho_calle = 2.0 / np.linalg.norm(w)
    sv_activos = np.sum(modelo['alpha'] > 1e-4)
    
    print("\n=== GEOMETRÍA EXTRAÍDA ===")
    print(f"Vector w: {w}")
    print(f"Sesgo b: {b:.4f}")
    print(f"Ancho del margen: {ancho_calle:.4f}")
    print(f"Vectores de soporte activos: {sv_activos} de {len(y)}")
    
    # Evaluar precisión
    y_pred = predecir_manual(X, modelo)
    exactitud = np.mean(y_pred == y) * 100
    print(f"Exactitud en la muestra: {exactitud:.1f}%")

    # Resumen de tiempos de cómputo
    t_total = time.perf_counter() - t_inicio_total
    print("\n=== TIEMPOS DE CÓMPUTO ===")
    print(f"Tiempo de carga y preparación de datos: {t_carga:.4f} s ({t_carga/60:.2f} min)")
    print(f"Tiempo de optimización (entrenamiento): {t_entrenamiento:.4f} s ({t_entrenamiento/60:.2f} min)")
    print(f"Tiempo total transcurrido:              {t_total:.4f} s ({t_total/60:.2f} min)")
