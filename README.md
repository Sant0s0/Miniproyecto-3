# 🎯 CS:GO Round Winner Prediction · SVM Lineal vs SVM RBF

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-2.35%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Academic-green)](#)

Sistema interactivo y modular de Machine Learning para la clasificación binaria y predicción del ganador de ronda en **Counter-Strike: Global Offensive (CS:GO)**, comparando el desempeño cuantitativo y cualitativo de **Máquinas de Vectores de Soporte (SVM)** con **Kernel Lineal** vs **Kernel RBF (Gaussiano)**.

Desarrollado para la asignatura de **Métodos de Inteligencia Artificial (Met-IA)** en la **Pontificia Universidad Javeriana Cali**.

---

## 📌 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Características Principales](#-características-principales)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Dataset](#-dataset)
- [Instalación y Requisitos](#-instalación-y-requisitos)
- [Guía de Uso](#-guía-de-uso)
  - [1. Aplicación Web Interactiva (Streamlit)](#1-aplicación-web-interactiva-streamlit)
  - [2. Scripts Independientes por Consola (CLI)](#2-scripts-independientes-por-consola-cli)
- [Comparación Metodológica: Lineal vs RBF](#-comparación-metodológica-lineal-vs-rbf)
  - [Formulación Matemática](#formulación-matemática)
  - [Tabla de Criterios Cualitativos](#tabla-de-criterios-cualitativos)
  - [Principio de Parsimonia (Veredicto)](#principio-de-parsimonia-veredicto)
- [Autores](#-autores)

---

## 📖 Descripción General

El objetivo del proyecto es modelar el estado competitivo de rondas de CS:GO tomadas en tiempo real para determinar con alta fidelidad si el bando ganador será **Terroristas (T = -1)** o **Counter-Terrorists (CT = +1)**. 

Se evalúan exhaustivamente dos enfoques basados en SVM:
1. **SVM Lineal (`LinearSVC`)**: Hiperplano óptimo en el espacio original de características, optimizando el margen blando con el parámetro de penalización $C$.
2. **SVM RBF (`SVC(kernel='rbf')`)**: Mapeo implícito a un espacio de Hilbert de dimensión infinita mediante el kernel Gaussiano, calibrando conjuntamente el margen $C$ y el ancho de banda del kernel $\gamma$.

---

## 🚀 Características Principales

* **Interfaz Web Profesional (Streamlit)**:
  * Diseño moderno *dark-glassmorphic* inspirado en interfaces de Esports.
  * Selector interactivo de hiperparámetros ($C$, $\gamma$), partición de datos modulable (80/20, 20/80, 50/50, etc.) y métrica de optimización en validación cruzada (`recall`, `accuracy`, `f1`, `roc_auc`).
* **Optimización de Entrenamiento en 2 Fases (Fast RBF GridSearch)**:
  * Resuelve la complejidad $\mathcal{O}(N^3)$ del kernel RBF calibrando hiperparámetros en una submuestra estratificada representativa y reentrenando el modelo final con el volumen total de entrenamiento.
  * Inferencia vectorizada única para eliminar tiempos muertos en evaluación.
* **Evaluación Cuantitativa Integral**:
  * 10 métricas de clasificación: Exactitud, Exactitud Balanceada, Precisión, Sensibilidad (Recall), Especificidad, F1-Score, Coeficiente de Correlación de Matthews (MCC), AUC-ROC, Precisión Promedio (AP) y Tasa de Error.
  * Perfilado de tiempos: GridSearchCV, ajuste medio por partición, refit y latencia de predicción en microsegundos ($\mu\text{s}$) por ronda.
  * Curvas interactivas ROC y Precisión-Recall, gráficos de validación cruzada y matrices de confusión con porcentajes normalizados.
* **Evaluación Cualitativa Avanzada**:
  * **Fronteras de decisión en 2D**: Proyección espacial mediante PCA (Componentes Principales) o selección directa de variables (`ct_armor` vs `t_armor`) visualizando el margen ($f(x) = \pm 1$) y los vectores de soporte.
  * **Distribución de $f(x)$**: Histogramas de certeza de decisión para T y CT con cálculo del porcentaje de rondas ambiguas en la zona de margen ($|f(x)| < 1$).
  * **Análisis automático y Veredicto**: Síntesis basada en la Navaja de Ockham evaluando si la no-linealidad justifica el costo computacional y de memoria.
* **Exportador de Informes Autónomos**:
  * Generación de reportes ejecutivos en HTML con todas las métricas, análisis y gráficos Plotly embebidos, con soporte optimizado para impresión directa y guardado en PDF (`@media print`).

---

## 📁 Estructura del Proyecto

```text
├── app_svm.py                  # Interfaz gráfica interactiva en Streamlit
├── svm_modelos.py              # Módulo central: preparación, entrenamiento 2-fases, métricas y fronteras 2D
├── csgo_svm_lineal_sklearn.py  # Script CLI independiente para SVM Lineal
├── csgo_svm_rbf_sklearn.py     # Script CLI independiente para SVM RBF optimizado
├── Kernel.py                   # Script base exploratorio de SVM
├── requirements.txt            # Dependencias del proyecto
├── .gitignore                  # Exclusión de venv, cache y archivos pesados
└── README.md                   # Documentación del repositorio
```

---

## 📊 Dataset

El proyecto utiliza el dataset **CS:GO Round Snapshots**:
* **Tamaño aproximado**: 122,410 registros correspondientes a instantáneas de partidas de torneos competitivos de alto nivel.
* **Dimensiones**: ~96 características numéricas que registran vida restante, armadura, dinero de equipo, armas de fuego disponibles (rifles, francotiradores, subfusiles), granadas, tiempo restante y bomba C4 plantada.
* **Variable objetivo (`round_winner`)**:
  * `CT`: Counter-Terrorists (codificado como `+1`).
  * `T`: Terrorists (codificado como `-1`).

> **Nota sobre los datos**: Debido al límite de tamaño de archivos en GitHub (>50MB), el archivo de datos (`csgo_round_snapshots.csv`) no se incluye en el control de versiones. Puedes descargarlo de [Kaggle - CS:GO Round Snapshots](https://www.kaggle.com/datasets/christianlillelund/csgo-round-snapshots) y colocarlo en la raíz del proyecto.

---

## 🛠️ Instalación y Requisitos

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/Sant0s0/Miniproyecto-3.git
   cd Miniproyecto-3
   ```

2. **Crear y activar un entorno virtual**:
   * En Windows (PowerShell / CMD):
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * En Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Guía de Uso

### 1. Aplicación Web Interactiva (Streamlit)

Para iniciar la interfaz gráfica completa con dashboard interactivo:

```bash
streamlit run app_svm.py
```

* Abre tu navegador en `http://localhost:8501`.
* Usa la barra lateral para ajustar el tamaño de muestra, semilla aleatoria, partición train/test (ej. 80/20 o 20/80) y rangos de hiperparámetros.
* Presiona **⚡ Entrenar ambos** para ejecutar los modelos y explorar las 4 pestañas:
  1. **📏 SVM Lineal**: Métricas detalladas, curvas, pesos $w$ del hiperplano y variables más influyentes.
  2. **🌀 SVM RBF**: Métricas, curvas, hiperparámetros óptimos y estadísticas de vectores de soporte.
  3. **⚖️ Comparación cuantitativa**: Tabla comparativa, gráficos de radar, barras de tiempos en escala logarítmica y botón de exportación de informe HTML/PDF.
  4. **🧭 Evaluación cualitativa**: Visualización de fronteras de decisión 2D (PCA o variables directas), distribución del margen $f(x)$ y análisis automático.

### 2. Scripts Independientes por Consola (CLI)

Si deseas ejecutar pruebas rápidas o entrenamientos por consola:

* **SVM Lineal**:
  ```bash
  python csgo_svm_lineal_sklearn.py
  ```
* **SVM RBF**:
  ```bash
  python csgo_svm_rbf_sklearn.py
  ```

---

## 🔬 Comparación Metodológica: Lineal vs RBF

### Formulación Matemática

* **SVM Lineal**:
  Encuentra el hiperplano $f(x) = w^T x + b = 0$ que maximiza el margen geométrico:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{sujeto a} \quad y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

* **SVM con Kernel RBF (Gaussiano)**:
  Aplica el truco del kernel para mapear datos no separables linealmente:
  $$K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)$$
  donde $\gamma$ controla el radio de influencia de cada vector de soporte.

### Tabla de Criterios Cualitativos

| Criterio | SVM Lineal (`LinearSVC`) | SVM RBF (`SVC(kernel='rbf')`) |
| :--- | :--- | :--- |
| **Forma de la frontera** | Hiperplano plano (recta en 2D, plano en 3D). | Curvas complejas, contornos no lineales y regiones no convexas. |
| **Interpretabilidad** | **Alta**: Vector explícito de pesos $w$ por variable (coeficientes). | **Baja (Caja negra)**: Decisión basada en similitud gaussiana con vectores de soporte. |
| **Escalabilidad ($N$ grande)** | **Excelente**: Complejidad aproximada $\mathcal{O}(N)$. | **Costosa**: Complejidad $\mathcal{O}(N^2)$ a $\mathcal{O}(N^3)$. |
| **Hiperparámetros a calibrar** | $C$ (penalización del margen blando). | $C$ y $\gamma$ (ancho de banda gaussiano). |
| **Memoria del modelo** | Solo almacena $96$ pesos $+ 1$ sesgo ($\sim 1 \text{ KB}$). | Requiere guardar miles de vectores de soporte completos en memoria ($\sim\text{MBs}$). |
| **Riesgo de sobreajuste** | Bajo / restringido por la linealidad. | Moderado a alto si $C$ o $\gamma$ son excesivamente elevados. |

### Principio de Parsimonia (Veredicto)

En el dominio de CS:GO, la mayoría de relaciones de ventaja (dinero acumulado, blindaje de equipo y vida total) presentan comportamientos monotónicos respecto a la probabilidad de victoria. Como resultado, la diferencia de exactitud entre el modelo RBF y el Lineal suele ser marginal ($\le \pm 0.5\%$).

Siguiendo el **Principio de Parsimonia (Navaja de Ockham)**, el **SVM Lineal es el modelo preferible para despliegues prácticos y de producción**:
* Es entre **$15\times$ y $20\times$ más rápido de calibrar**.
* La latencia de inferencia por ronda es del orden de unidades de microsegundos ($\sim 2.5\,\mu\text{s}$ vs $\sim 600\,\mu\text{s}$ en RBF).
* Permite explicabilidad inmediata al cuerpo técnico y jugadores analizando qué armas o recursos inclinan la balanza de la ronda.

---

## 👥 Autores

Proyecto desarrollado por:
* **Juan Alejandro Cardona Morales** ([@Sant0s0](https://github.com/Sant0s0))
* Pontificia Universidad Javeriana Cali — Facultad de Ingeniería y Ciencias
* Asignatura: Métodos de Inteligencia Artificial (Met-IA)
