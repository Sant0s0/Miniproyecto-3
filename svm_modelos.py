"""
Núcleo compartido para comparar SVM Lineal vs SVM con kernel RBF
sobre el dataset CS:GO Round Snapshots.

Lo usan:
    - app_svm.py               (interfaz gráfica Streamlit)
    - csgo_svm_rbf_sklearn.py  (script independiente RBF)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

RUTA_DATASET = Path(__file__).with_name("csgo_round_snapshots.csv")
NOMBRES_CLASES = ["Terrorista (-1)", "Antiterrorista (+1)"]


# ----------------------------------------------------------------------------
# 1. Datos
# ----------------------------------------------------------------------------
def cargar_dataset(filepath=RUTA_DATASET) -> pd.DataFrame:
    """Lee el CSV y lo deja listo para clasificación binaria (CT=+1, T=-1)."""
    df = pd.read_csv(filepath)
    df["round_winner"] = df["round_winner"].map({"CT": 1, "T": -1})
    df = df.dropna(subset=["round_winner"])
    df["round_winner"] = df["round_winner"].astype(int)

    # bomb_planted viene como True/False
    if "bomb_planted" in df.columns:
        df["bomb_planted"] = df["bomb_planted"].astype(str).str.lower().eq("true").astype(int)

    categoricas = [c for c in df.select_dtypes(include=["object", "category"]).columns
                   if c != "round_winner"]
    df = pd.get_dummies(df, columns=categoricas, drop_first=True, dtype=int)
    return df


def preparar_datos(df, n_muestras=None, test_size=0.20, random_state=42):
    """Submuestreo estratificado opcional + partición train/test estratificada."""
    X = df.drop(columns=["round_winner"]).astype(float)
    y = df["round_winner"].values

    if n_muestras is not None and n_muestras < len(df):
        X, _, y, _ = train_test_split(
            X, y, train_size=n_muestras, stratify=y, random_state=random_state
        )
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


# ----------------------------------------------------------------------------
# 2. Entrenamiento (calibración de hiperparámetros)
# ----------------------------------------------------------------------------
def construir_busqueda(tipo, Cs, gammas=None, cv=5, scoring="recall",
                       n_jobs=-1, random_state=42):
    """Pipeline (StandardScaler + SVM) dentro de un GridSearchCV."""
    if tipo == "lineal":
        estimador = LinearSVC(dual=False, random_state=random_state, max_iter=5000)
        grid = {"svm__C": list(Cs)}
    elif tipo == "rbf":
        estimador = SVC(kernel="rbf", cache_size=1000)
        grid = {"svm__C": list(Cs), "svm__gamma": list(gammas or ["scale"])}
    else:
        raise ValueError(f"Tipo de modelo desconocido: {tipo}")

    pipe = Pipeline([("scaler", StandardScaler()), ("svm", estimador)])
    particiones = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return GridSearchCV(pipe, grid, cv=particiones, scoring=scoring,
                        n_jobs=n_jobs, return_train_score=True)


def entrenar_modelo(tipo, X_train, y_train, n_submuestra_busqueda=None, **kwargs):
    """
    Devuelve (GridSearchCV ajustado, segundos de calibración).
    Si n_submuestra_busqueda se especifica (p. ej. 2000) y len(X_train) > n_submuestra_busqueda:
        1. Calibra C y gamma rápidamente en la submuestra estratificada.
        2. Re-entrena el mejor modelo (best_estimator_) sobre la totalidad de X_train.
        Esto reduce el tiempo de calibración de minutos/horas a segundos.
    """
    busqueda = construir_busqueda(tipo, **kwargs)
    random_state = kwargs.get("random_state", 42)

    if n_submuestra_busqueda is not None and len(X_train) > n_submuestra_busqueda:
        X_busq, _, y_busq, _ = train_test_split(
            X_train, y_train, train_size=n_submuestra_busqueda,
            stratify=y_train, random_state=random_state
        )
        t0 = time.perf_counter()
        busqueda.fit(X_busq, y_busq)
        t_grid = time.perf_counter() - t0

        # Re-entrenamiento del mejor estimador con todo X_train
        t0_refit = time.perf_counter()
        busqueda.best_estimator_.fit(X_train, y_train)
        busqueda.refit_time_ = time.perf_counter() - t0_refit
        busqueda.n_submuestra_ = n_submuestra_busqueda
    else:
        t0 = time.perf_counter()
        busqueda.fit(X_train, y_train)
        t_grid = time.perf_counter() - t0
        busqueda.n_submuestra_ = len(X_train)

    return busqueda, t_grid


def resumen_cv(busqueda) -> pd.DataFrame:
    """Tabla con los resultados de la validación cruzada por combinación."""
    res = pd.DataFrame(busqueda.cv_results_)
    cols_param = [c for c in res.columns if c.startswith("param_")]
    cols = cols_param + ["mean_test_score", "std_test_score", "mean_train_score",
                         "mean_fit_time", "rank_test_score"]
    out = res[cols].copy()
    out.columns = [c.replace("param_svm__", "") for c in out.columns]
    out["C"] = out["C"].astype(float)
    if "gamma" in out.columns:
        out["gamma"] = out["gamma"].astype(str)
    return out


# ----------------------------------------------------------------------------
# 3. Evaluación cuantitativa
# ----------------------------------------------------------------------------
def evaluar_modelo(busqueda, X_test, y_test):
    """Calcula métricas, matriz de confusión, curvas ROC/PR y tiempos de predicción."""
    t0 = time.perf_counter()
    puntajes = busqueda.decision_function(X_test)
    y_pred = np.where(puntajes >= 0, 1, -1)
    t_pred = time.perf_counter() - t0

    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_test, puntajes, pos_label=1)
    prec, rec, _ = precision_recall_curve(y_test, puntajes, pos_label=1)
    exactitud = accuracy_score(y_test, y_pred)

    metricas = {
        "Exactitud": exactitud,
        "Tasa de error": 1 - exactitud,
        "Precisión": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Sensibilidad (Recall)": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Exactitud balanceada": balanced_accuracy_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, puntajes),
        "Precisión promedio (AP)": average_precision_score(y_test, puntajes, pos_label=1),
    }

    svm = busqueda.best_estimator_.named_steps["svm"]
    extras = {"frac_zona_margen": float(np.mean(np.abs(puntajes) < 1))}
    if isinstance(svm, SVC):
        extras["n_vectores_soporte"] = int(svm.n_support_.sum())
        extras["vectores_por_clase"] = svm.n_support_.tolist()  # [T, CT]
    else:
        extras["coeficientes"] = pd.Series(svm.coef_.ravel(), index=X_test.columns)
        extras["intercepto"] = float(svm.intercept_[0])

    reporte = classification_report(y_test, y_pred, labels=[-1, 1],
                                    target_names=NOMBRES_CLASES,
                                    output_dict=True, zero_division=0)

    return {
        "metricas": metricas,
        "cm": cm,
        "tn_fp_fn_tp": (int(tn), int(fp), int(fn), int(tp)),
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"precision": prec, "recall": rec, "base": float(np.mean(y_test == 1))},
        "puntajes": puntajes,
        "y_test": np.asarray(y_test),
        "y_pred": y_pred,
        "t_pred": t_pred,
        "reporte": reporte,
        "extras": extras,
    }


# ----------------------------------------------------------------------------
# 4. Evaluación cualitativa: frontera de decisión en 2D
# ----------------------------------------------------------------------------
def frontera_decision_2d(tipo, mejores_params, X_train, y_train, modo="pca",
                         variables=None, n_max=1500, resolucion=150,
                         gamma_2d_scale=True, random_state=42):
    """
    Proyecta los datos a 2D (PCA o 2 variables), re-entrena el SVM con los
    mejores hiperparámetros y evalúa la función de decisión en una malla.
    """
    y_train = np.asarray(y_train)
    idx = np.arange(len(y_train))
    if len(idx) > n_max:
        idx, _ = train_test_split(idx, train_size=n_max, stratify=y_train,
                                  random_state=random_state)
    Xs, ys = X_train.iloc[idx], y_train[idx]

    if modo == "pca":
        escalador = StandardScaler().fit(X_train)
        pca = PCA(n_components=2, random_state=random_state).fit(escalador.transform(X_train))
        X2 = pca.transform(escalador.transform(Xs))
        var = pca.explained_variance_ratio_
        ejes = (f"PC1 ({var[0] * 100:.1f}% var.)", f"PC2 ({var[1] * 100:.1f}% var.)")
    else:
        X2 = StandardScaler().fit_transform(Xs[list(variables)].values)
        ejes = (f"{variables[0]} (estandarizada)", f"{variables[1]} (estandarizada)")

    params = {k.replace("svm__", ""): v for k, v in mejores_params.items()}
    if tipo == "lineal":
        clf = LinearSVC(dual=False, max_iter=5000, random_state=random_state, **params)
    else:
        if gamma_2d_scale:
            params["gamma"] = "scale"
        clf = SVC(kernel="rbf", **params)

    t0 = time.perf_counter()
    clf.fit(X2, ys)
    t_fit = time.perf_counter() - t0

    pad = 0.5
    xs = np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + pad, resolucion)
    yy = np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + pad, resolucion)
    malla_x, malla_y = np.meshgrid(xs, yy)
    Z = clf.decision_function(np.c_[malla_x.ravel(), malla_y.ravel()]).reshape(malla_x.shape)

    f_train = clf.decision_function(X2)
    if isinstance(clf, SVC):
        sv_idx = clf.support_
    else:  # en el lineal: puntos sobre o dentro del margen (y*f(x) <= 1)
        sv_idx = np.where(ys * f_train <= 1)[0]

    return {
        "xs": xs, "ys_grid": yy, "Z": Z, "X2": X2, "y": ys,
        "ejes": ejes, "t_fit": t_fit, "acc_2d": float(clf.score(X2, ys)),
        "sv_idx": sv_idx, "params": params,
    }
