"""
Interfaz gráfica: SVM Lineal vs SVM RBF — CS:GO Round Snapshots
Ejecutar con:   streamlit run app_svm.py
"""
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import svm_modelos as sm

# ----------------------------------------------------------------------------
# Configuración general y estilos
# ----------------------------------------------------------------------------
st.set_page_config(page_title="SVM Lineal vs RBF · CS:GO", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

COLORES = {"lineal": "#22d3ee", "rbf": "#f472b6"}
NOMBRES = {"lineal": "SVM Lineal", "rbf": "SVM RBF"}
COLOR_T, COLOR_CT = "#f59e0b", "#60a5fa"
PALETA = ["#f472b6", "#a78bfa", "#facc15", "#34d399", "#fb923c", "#38bdf8"]
METRICAS_PRINCIPALES = ["Exactitud", "Precisión", "Sensibilidad (Recall)",
                        "Especificidad", "F1", "AUC-ROC"]
SCORINGS = {"recall": "Sensibilidad (Recall)", "accuracy": "Exactitud", "f1": "F1",
            "roc_auc": "AUC-ROC", "balanced_accuracy": "Exactitud balanceada"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #1e1b4b 0%, #0b1020 45%, #050814 100%); }
section[data-testid="stSidebar"] { background: rgba(10,14,30,.85); border-right: 1px solid rgba(255,255,255,.06); }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }
.hero { padding: 1.6rem 2rem; border-radius: 22px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(34,211,238,.14), rgba(167,139,250,.10), rgba(244,114,182,.14));
        border: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(14px); animation: fadeIn .6s ease; }
.hero h1 { font-weight: 800; font-size: 2.3rem; margin: 0; padding: 0;
           background: linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #cbd5e1; margin: .4rem 0 0 0; }
.card { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px; padding: .9rem 1.1rem; margin-bottom: .6rem;
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease; animation: fadeIn .5s ease; }
.card:hover { transform: translateY(-3px); border-color: rgba(167,139,250,.55);
              box-shadow: 0 10px 30px -12px rgba(167,139,250,.45); }
.card .lbl { color: #94a3b8; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
.card .val { font-family: 'Outfit'; font-size: 1.65rem; font-weight: 700; color: #f8fafc; line-height: 1.2; }
.card .sub { font-size: .78rem; color: #64748b; }
.badge { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .78rem;
         font-weight: 600; margin: 0 .35rem .35rem 0; border: 1px solid rgba(255,255,255,.12); }
.nota { background: rgba(167,139,250,.08); border-left: 3px solid #a78bfa; border-radius: 10px;
        padding: .8rem 1rem; color: #cbd5e1; font-size: .9rem; margin: .4rem 0 1rem 0; }
.stTabs [data-baseweb="tab"] { font-family: 'Outfit'; font-weight: 600; font-size: 1rem; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
</style>
""", unsafe_allow_html=True)

if "resultados" not in st.session_state:
    st.session_state.resultados = {}
if "fronteras" not in st.session_state:
    st.session_state.fronteras = {}
RES = st.session_state.resultados


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def fmt_t(s):
    if s < 1:
        return f"{s * 1000:.1f} ms"
    if s < 60:
        return f"{s:.2f} s"
    return f"{s / 60:.1f} min"


def tarjeta(label, valor, sub="", color="#a78bfa"):
    return (f'<div class="card" style="border-left:4px solid {color}">'
            f'<div class="lbl">{label}</div><div class="val">{valor}</div>'
            f'<div class="sub">{sub}</div></div>')


def fila_tarjetas(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(tarjeta(*it), unsafe_allow_html=True)


def estilo(fig, titulo=None, alto=380):
    kw = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(255,255,255,0.02)", height=alto,
              font=dict(family="Inter", color="#e2e8f0"),
              margin=dict(l=50, r=20, t=60 if titulo else 25, b=50),
              legend=dict(orientation="h", y=-0.22, x=0))
    if titulo:
        kw["title"] = dict(text=titulo, font=dict(family="Outfit", size=17))
    fig.update_layout(**kw)
    return fig


def grafico(fig, key):
    st.plotly_chart(fig, key=key, config={"displaylogo": False})


@st.cache_data(show_spinner=False)
def cargar():
    t0 = time.perf_counter()
    df = sm.cargar_dataset()
    return df, time.perf_counter() - t0


@st.cache_data(show_spinner=False)
def preparar(n, test_size, seed):
    df, _ = cargar()
    t0 = time.perf_counter()
    partes = sm.preparar_datos(df, n, test_size, seed)
    return partes, time.perf_counter() - t0


# ----------------------------------------------------------------------------
# Gráficas
# ----------------------------------------------------------------------------
def fig_confusion(tipo, titulo=None):
    cm = RES[tipo]["eval"]["cm"]
    z = cm / cm.sum(axis=1, keepdims=True)
    texto = [[f"<b>{cm[i, j]:,}</b><br>{z[i, j] * 100:.1f}%" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicho T (−1)", "Predicho CT (+1)"], y=["Real T (−1)", "Real CT (+1)"],
        text=texto, texttemplate="%{text}", textfont=dict(size=16),
        colorscale=[[0, "#0f172a"], [1, COLORES[tipo]]], zmin=0, zmax=1, showscale=False,
        hovertemplate="%{y} → %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return estilo(fig, titulo or f"Matriz de confusión · {NOMBRES[tipo]}", 360)


def fig_roc(tipos):
    fig = go.Figure()
    for t in tipos:
        ev = RES[t]["eval"]
        fig.add_trace(go.Scatter(
            x=ev["roc"]["fpr"], y=ev["roc"]["tpr"], mode="lines",
            name=f"{NOMBRES[t]} (AUC = {ev['metricas']['AUC-ROC']:.3f})",
            line=dict(color=COLORES[t], width=3),
            fill="tozeroy" if len(tipos) == 1 else None))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                             line=dict(color="#64748b", dash="dash")))
    fig.update_xaxes(title="Tasa de falsos positivos (1 − especificidad)", range=[0, 1])
    fig.update_yaxes(title="Tasa de verdaderos positivos (recall)", range=[0, 1.02])
    return estilo(fig, "Curva ROC", 380)


def fig_pr(tipos):
    fig = go.Figure()
    for t in tipos:
        ev = RES[t]["eval"]
        fig.add_trace(go.Scatter(
            x=ev["pr"]["recall"], y=ev["pr"]["precision"], mode="lines",
            name=f"{NOMBRES[t]} (AP = {ev['metricas']['Precisión promedio (AP)']:.3f})",
            line=dict(color=COLORES[t], width=3)))
    base = RES[tipos[0]]["eval"]["pr"]["base"]
    fig.add_hline(y=base, line_dash="dash", line_color="#64748b",
                  annotation_text=f"Línea base ({base:.2f})")
    fig.update_xaxes(title="Recall (sensibilidad)", range=[0, 1])
    fig.update_yaxes(title="Precisión", range=[0, 1.02])
    return estilo(fig, "Curva Precisión-Recall", 380)


def fig_validacion(tipo):
    r = RES[tipo]
    df = r["cv"]
    fig = go.Figure()
    grupos = [(None, df)] if "gamma" not in df.columns else list(df.groupby("gamma"))
    for i, (g, d) in enumerate(grupos):
        d = d.sort_values("C")
        nombre = "" if g is None else f"γ={g} · "
        color = COLORES[tipo] if g is None else PALETA[i % len(PALETA)]
        fig.add_trace(go.Scatter(
            x=d["C"], y=d["mean_test_score"], mode="lines+markers", name=f"{nombre}validación",
            error_y=dict(type="data", array=d["std_test_score"], visible=True, thickness=1),
            line=dict(color=color, width=3), marker=dict(size=9)))
        fig.add_trace(go.Scatter(
            x=d["C"], y=d["mean_train_score"], mode="lines", name=f"{nombre}entrenamiento",
            line=dict(color=color, width=1.5, dash="dot"), opacity=.7))
    mejor_c = r["mejores"]["svm__C"]
    y_min = float(min(df["mean_test_score"].min(), df["mean_train_score"].min()))
    y_max = float(max(df["mean_test_score"].max(), df["mean_train_score"].max()))
    fig.add_trace(go.Scatter(x=[mejor_c, mejor_c], y=[y_min, y_max], mode="lines+text",
                             name=f"Mejor C = {mejor_c}", text=["", f"Mejor C = {mejor_c}"],
                             textposition="top center", textfont=dict(color="#facc15"),
                             line=dict(color="#facc15", dash="dash", width=2)))
    fig.update_xaxes(type="log", title="C (margen blando, escala log)")
    fig.update_yaxes(title=f"{SCORINGS.get(r['config']['scoring'], r['config']['scoring'])} (CV)")
    return estilo(fig, "Curva de validación de C", 380)


def fig_tiempos_modelo(tipo):
    t = RES[tipo]["tiempos"]
    etiquetas = ["Preparación / partición", "Búsqueda GridSearchCV (total)",
                 "Ajuste medio por fold", "Re-entrenamiento final (refit)", "Predicción en test"]
    valores = [t["preparacion"], t["gridsearch"], t["fit_medio"], t["refit"], t["prediccion"]]
    fig = go.Figure(go.Bar(
        x=valores, y=etiquetas, orientation="h", marker_color=COLORES[tipo],
        text=[fmt_t(v) for v in valores], textposition="outside"))
    fig.update_xaxes(title="segundos", type="log")
    fig.update_yaxes(autorange="reversed")
    return estilo(fig, "Desglose de tiempos de cómputo (escala log)", 340)


def fig_coeficientes(n=15):
    coef = RES["lineal"]["eval"]["extras"]["coeficientes"]
    top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:n][::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=[COLOR_CT if v > 0 else COLOR_T for v in top.values],
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#64748b")
    fig.update_xaxes(title="Peso w (azul favorece CT · ámbar favorece T)")
    return estilo(fig, f"Top {n} variables más influyentes (pesos del hiperplano)", 460)


def fig_distribucion_margen(tipo):
    ev = RES[tipo]["eval"]
    s, y = ev["puntajes"], ev["y_test"]
    fig = go.Figure()
    for clase, nombre, color in [(-1, "Real T", COLOR_T), (1, "Real CT", COLOR_CT)]:
        fig.add_trace(go.Histogram(x=s[y == clase], name=nombre, marker_color=color,
                                   opacity=.65, nbinsx=60))
    for v, txt in [(-1, "−1"), (0, "f(x)=0"), (1, "+1")]:
        fig.add_vline(x=v, line_dash="solid" if v == 0 else "dot",
                      line_color="#f8fafc" if v == 0 else "#94a3b8", annotation_text=txt)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Valor de la función de decisión f(x)")
    fig.update_yaxes(title="Nº de rondas (test)")
    return estilo(fig, f"Distribución de f(x) · {NOMBRES[tipo]}", 360)


def fig_frontera(tipo, fr):
    Z = np.clip(fr["Z"], -3, 3)
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=fr["xs"], y=fr["ys_grid"], z=Z, zmin=-3, zmax=3, showscale=False, opacity=.75,
        colorscale=[[0, COLOR_T], [0.5, "#0f172a"], [1, COLOR_CT]],
        contours=dict(start=-3, end=3, size=0.5), line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Contour(
        x=fr["xs"], y=fr["ys_grid"], z=fr["Z"], showscale=False, hoverinfo="skip",
        contours=dict(start=-1, end=1, size=1, coloring="lines", showlabels=True,
                      labelfont=dict(color="white")),
        colorscale=[[0, "#ffffff"], [1, "#ffffff"]], line=dict(width=2, dash="dot")))
    X2, y = fr["X2"], fr["y"]
    for clase, nombre, color in [(-1, "T (−1)", COLOR_T), (1, "CT (+1)", COLOR_CT)]:
        m = y == clase
        fig.add_trace(go.Scatter(x=X2[m, 0], y=X2[m, 1], mode="markers", name=nombre,
                                 marker=dict(color=color, size=6, line=dict(width=.5, color="#0f172a"))))
    sv = fr["sv_idx"]
    etiqueta_sv = "Vectores de soporte" if tipo == "rbf" else "En el margen (y·f(x) ≤ 1)"
    fig.add_trace(go.Scatter(x=X2[sv, 0], y=X2[sv, 1], mode="markers", name=etiqueta_sv,
                             marker=dict(symbol="circle-open", size=10, color="#f8fafc",
                                         line=dict(width=1)), opacity=.5))
    fig.update_xaxes(title=fr["ejes"][0])
    fig.update_yaxes(title=fr["ejes"][1])
    return estilo(fig, f"Frontera de decisión 2D · {NOMBRES[tipo]}", 520)


def fig_radar():
    L, R = RES["lineal"], RES["rbf"]
    mL, mR = L["eval"]["metricas"], R["eval"]["metricas"]
    fig = go.Figure()
    for t, r in [("lineal", mL), ("rbf", mR)]:
        vals = [r[k] for k in METRICAS_PRINCIPALES]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=METRICAS_PRINCIPALES + METRICAS_PRINCIPALES[:1],
            fill="toself", name=NOMBRES[t], line=dict(color=COLORES[t], width=2), opacity=.75
        ))
    mn = min(min(mL[k], mR[k]) for k in METRICAS_PRINCIPALES)
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[max(0, mn - 0.05), 1], gridcolor="#334155")))
    return estilo(fig, "Perfil de desempeño (radar)", 460)


def fig_barras_metricas():
    L, R = RES["lineal"], RES["rbf"]
    mL, mR = L["eval"]["metricas"], R["eval"]["metricas"]
    mn = min(min(mL[k], mR[k]) for k in METRICAS_PRINCIPALES)
    fig = go.Figure()
    for t, r in [("lineal", mL), ("rbf", mR)]:
        fig.add_trace(go.Bar(
            x=METRICAS_PRINCIPALES, y=[r[k] for k in METRICAS_PRINCIPALES],
            name=NOMBRES[t], marker_color=COLORES[t],
            text=[f"{r[k]:.3f}" for k in METRICAS_PRINCIPALES], textposition="outside"
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[max(0, mn - 0.1), 1.05])
    return estilo(fig, "Métricas de clasificación", 400)


def fig_barras_tiempos():
    L, R = RES["lineal"], RES["rbf"]
    tL, tR = L["tiempos"], R["tiempos"]
    etiquetas = ["GridSearchCV", "Ajuste medio/fold", "Refit", "Predicción"]
    claves = ["gridsearch", "fit_medio", "refit", "prediccion"]
    fig = go.Figure()
    for t, tt in [("lineal", tL), ("rbf", tR)]:
        fig.add_trace(go.Bar(
            x=etiquetas, y=[tt[k] for k in claves], name=NOMBRES[t],
            marker_color=COLORES[t], text=[fmt_t(tt[k]) for k in claves],
            textposition="outside"
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(type="log", title="segundos (escala log)")
    return estilo(fig, "Tiempos de cómputo", 400)


def analisis_automatico():
    L, R = RES["lineal"], RES["rbf"]
    mL, mR = L["eval"]["metricas"], R["eval"]["metricas"]
    tL, tR = L["tiempos"], R["tiempos"]
    (_, fpL, fnL, _), (_, fpR, fnR, _) = L["eval"]["tn_fp_fn_tp"], R["eval"]["tn_fp_fn_tp"]
    d_acc = (mR["Exactitud"] - mL["Exactitud"]) * 100
    ratio = tR["gridsearch"] / max(tL["gridsearch"], 1e-9)
    zL, zR = L["eval"]["extras"]["frac_zona_margen"], R["eval"]["extras"]["frac_zona_margen"]
    nsv = R["eval"]["extras"]["n_vectores_soporte"]

    lineas = [
        f"- **Exactitud:** el RBF obtiene {mR['Exactitud'] * 100:.2f}% frente a "
        f"{mL['Exactitud'] * 100:.2f}% del lineal (**{d_acc:+.2f} pp**).",
        f"- **Costo computacional:** la calibración del RBF tardó **{ratio:.1f}×** lo del lineal "
        f"({fmt_t(tR['gridsearch'])} vs {fmt_t(tL['gridsearch'])}); predecir una ronda cuesta "
        f"{tR['pred_por_muestra'] * 1e6:.1f} µs vs {tL['pred_por_muestra'] * 1e6:.1f} µs.",
        f"- **Tipos de error:** falsos positivos (se predijo CT pero ganó T): lineal {fpL:,} vs RBF {fpR:,}; "
        f"falsos negativos (se predijo T pero ganó CT): lineal {fnL:,} vs RBF {fnR:,}.",
        f"- **Confianza / separabilidad:** el {zL * 100:.1f}% (lineal) y el {zR * 100:.1f}% (RBF) de las "
        f"rondas de prueba caen dentro del margen (|f(x)| < 1); un valor menor indica clases más separadas.",
        f"- **Sobreajuste:** brecha entrenamiento − validación del mejor modelo: lineal "
        f"{L['gap_train_val']:+.4f}, RBF {R['gap_train_val']:+.4f}.",
        f"- **Complejidad / memoria:** el lineal se resume en un vector w de {len(COLUMNAS)} pesos + sesgo; "
        f"el RBF necesita guardar **{nsv:,} vectores de soporte** "
        f"({nsv / R['n_train'] * 100:.1f}% del entrenamiento).",
    ]
    if d_acc > 1.0:
        concl = (f"El kernel RBF captura relaciones **no lineales** que mejoran la exactitud en {d_acc:.2f} pp; "
                 f"compensa su mayor costo si la precisión es prioritaria.")
    elif d_acc < -1.0:
        concl = "El modelo lineal es **más exacto y más barato**: el RBF no aporta ventaja en esta configuración."
    else:
        concl = (f"La diferencia de exactitud es pequeña ({d_acc:+.2f} pp). Por **parsimonia**, el modelo "
                 f"lineal resulta preferible: es {ratio:.1f}× más rápido de calibrar e interpretable.")
    return "\n".join(lineas), concl


def generar_informe_html():
    """Genera un archivo HTML autónomo e interactivo con métricas, tablas, análisis y gráficas Plotly."""
    entrenados = [t for t in ("lineal", "rbf") if t in RES]
    if not entrenados:
        return "<!DOCTYPE html><html><body><h1>No hay modelos entrenados aún.</h1></body></html>"

    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = RES[entrenados[0]]["config"]
    n_train = RES[entrenados[0]]["n_train"]
    n_test = RES[entrenados[0]]["n_test"]

    # 1. Tarjetas KPI
    tarjetas_html = ""
    for t in entrenados:
        r = RES[t]
        m = r["eval"]["metricas"]
        tiem = r["tiempos"]
        color = COLORES[t]
        params_str = ", ".join(f"{k.replace('svm__', '')}={v}" for k, v in r["mejores"].items())
        tarjetas_html += f"""
        <div class="kpi-card" style="border-top: 4px solid {color};">
            <div class="kpi-title">{NOMBRES[t]}</div>
            <div class="kpi-val" style="color: {color};">{m['Exactitud'] * 100:.2f}%</div>
            <div class="kpi-sub">F1: {m['F1']:.3f} | AUC: {m['AUC-ROC']:.3f}</div>
            <div class="kpi-meta">
                <b>Hiperparámetros:</b> {params_str}<br>
                <b>Tiempo total:</b> {fmt_t(tiem['total'])} (Predicción: {fmt_t(tiem['prediccion'])})
            </div>
        </div>
        """

    # 2. Tabla de métricas
    if len(entrenados) == 2:
        L, R = RES["lineal"], RES["rbf"]
        mL, mR = L["eval"]["metricas"], R["eval"]["metricas"]
        tL, tR = L["tiempos"], R["tiempos"]
        filas = []
        for k in mL:
            menor_es_mejor = (k == "Tasa de error")
            mejor = "Lineal" if (mL[k] < mR[k]) == menor_es_mejor else "RBF"
            if abs(mL[k] - mR[k]) < 1e-4:
                mejor = "Empate"
            filas.append((k, f"{mL[k]:.4f}", f"{mR[k]:.4f}", f"{mR[k] - mL[k]:+.4f}", mejor))
        for nombre, clave in [("Tiempo GridSearch (s)", "gridsearch"), ("Tiempo refit (s)", "refit"),
                              ("Tiempo predicción (s)", "prediccion"), ("Tiempo total (s)", "total")]:
            mejor = "Lineal" if tL[clave] < tR[clave] else "RBF"
            filas.append((nombre, f"{tL[clave]:.3f}", f"{tR[clave]:.3f}", f"{tR[clave] - tL[clave]:+.3f}", mejor))

        filas_tr = "".join(f"""
            <tr>
                <td><b>{f[0]}</b></td>
                <td style="color:{COLORES['lineal']};">{f[1]}</td>
                <td style="color:{COLORES['rbf']};">{f[2]}</td>
                <td>{f[3]}</td>
                <td><span class="badge badge-{f[4].lower()}">{f[4]}</span></td>
            </tr>
        """ for f in filas)

        tabla_html = f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th>Métrica / Criterio</th>
                    <th>SVM Lineal</th>
                    <th>SVM RBF</th>
                    <th>Δ (RBF − Lineal)</th>
                    <th>Mejor</th>
                </tr>
            </thead>
            <tbody>
                {filas_tr}
            </tbody>
        </table>
        """
    else:
        t = entrenados[0]
        m = RES[t]["eval"]["metricas"]
        filas_tr = "".join(f"<tr><td><b>{k}</b></td><td>{v:.4f}</td></tr>" for k, v in m.items())
        tabla_html = f"""
        <table class="report-table">
            <thead><tr><th>Métrica</th><th>Valor</th></tr></thead>
            <tbody>{filas_tr}</tbody>
        </table>
        """

    # 3. Gráficas interactivas Plotly
    cfg_plotly = {"responsive": True, "displayModeBar": False}
    if len(entrenados) == 2:
        g_radar = fig_radar().to_html(full_html=False, include_plotlyjs=False, div_id="chart_radar", config=cfg_plotly)
        g_barras_m = fig_barras_metricas().to_html(full_html=False, include_plotlyjs=False, div_id="chart_barras_m", config=cfg_plotly)
        g_barras_t = fig_barras_tiempos().to_html(full_html=False, include_plotlyjs=False, div_id="chart_barras_t", config=cfg_plotly)
        g_roc = fig_roc(["lineal", "rbf"]).to_html(full_html=False, include_plotlyjs=False, div_id="chart_roc", config=cfg_plotly)
        g_pr = fig_pr(["lineal", "rbf"]).to_html(full_html=False, include_plotlyjs=False, div_id="chart_pr", config=cfg_plotly)
        g_dist = fig_distribucion_margen("rbf").to_html(full_html=False, include_plotlyjs=False, div_id="chart_dist", config=cfg_plotly)
        g_cm_l = fig_confusion("lineal").to_html(full_html=False, include_plotlyjs=False, div_id="chart_cm_l", config=cfg_plotly)
        g_cm_r = fig_confusion("rbf").to_html(full_html=False, include_plotlyjs=False, div_id="chart_cm_r", config=cfg_plotly)

        graficos_html = f"""
        <div class="chart-row">
            <div class="chart-card">{g_radar}</div>
            <div class="chart-card">{g_barras_m}</div>
        </div>
        <div class="chart-row">
            <div class="chart-card">{g_barras_t}</div>
            <div class="chart-card">{g_roc}</div>
        </div>
        <div class="chart-row">
            <div class="chart-card">{g_pr}</div>
            <div class="chart-card">{g_dist}</div>
        </div>
        <div class="chart-row">
            <div class="chart-card">{g_cm_l}</div>
            <div class="chart-card">{g_cm_r}</div>
        </div>
        """
    else:
        t = entrenados[0]
        g_roc = fig_roc([t]).to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_roc_{t}", config=cfg_plotly)
        g_pr = fig_pr([t]).to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_pr_{t}", config=cfg_plotly)
        g_cm = fig_confusion(t).to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_cm_{t}", config=cfg_plotly)
        g_val = fig_validacion(t).to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_val_{t}", config=cfg_plotly)
        graficos_html = f"""
        <div class="chart-row">
            <div class="chart-card">{g_roc}</div>
            <div class="chart-card">{g_pr}</div>
        </div>
        <div class="chart-row">
            <div class="chart-card">{g_cm}</div>
            <div class="chart-card">{g_val}</div>
        </div>
        """

    # 4. Análisis automático
    analisis_html = ""
    if len(entrenados) == 2:
        lineas, concl = analisis_automatico()
        items_p = []
        for linea in lineas.split("\n"):
            linea = linea.strip()
            if not linea:
                continue
            if linea.startswith("- "):
                linea = linea[2:]
            linea_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", linea)
            items_p.append(f'<p style="margin-bottom: 8px;">• {linea_html}</p>')
        items_html = "\n".join(items_p)
        concl_html = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", concl)
        analisis_html = f"""
        <div class="analysis-box">
            <h3>🧠 Conclusión y Análisis Automático</h3>
            <div style="line-height: 1.7; color: #cbd5e1; margin-bottom: 15px;">
                {items_html}
            </div>
            <div class="concl-alert"><b>Veredicto:</b> {concl_html}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe Comparativo SVM: Lineal vs RBF · CS:GO</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0b1020;
            color: #e2e8f0;
            padding: 2.5rem;
            line-height: 1.5;
        }}
        h1, h2, h3, h4 {{ font-family: 'Outfit', sans-serif; font-weight: 700; }}
        .header {{
            background: linear-gradient(135deg, rgba(34,211,238,0.12), rgba(167,139,250,0.12), rgba(244,114,182,0.12));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .header h1 {{
            font-size: 2.4rem;
            background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header p {{ color: #94a3b8; font-size: 0.95rem; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .kpi-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1.4rem;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .kpi-title {{ font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.08em; }}
        .kpi-val {{ font-family: 'Outfit'; font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; }}
        .kpi-sub {{ font-size: 0.88rem; color: #cbd5e1; }}
        .kpi-meta {{ font-size: 0.78rem; color: #64748b; margin-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.6rem; }}
        .section-title {{
            font-size: 1.4rem;
            color: #f8fafc;
            margin: 2rem 0 1rem 0;
            border-left: 4px solid #a78bfa;
            padding-left: 0.75rem;
            page-break-after: avoid;
            break-after: avoid;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .report-table th, .report-table td {{
            padding: 0.85rem 1.2rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            font-size: 0.92rem;
        }}
        .report-table th {{
            background: rgba(255,255,255,0.05);
            font-family: 'Outfit';
            color: #f1f5f9;
            font-weight: 600;
        }}
        .report-table tr:hover {{ background: rgba(255,255,255,0.03); }}
        .report-table tr {{ page-break-inside: avoid; break-inside: avoid; }}
        .badge {{
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-lineal {{ background: rgba(34,211,238,0.2); color: #22d3ee; border: 1px solid rgba(34,211,238,0.4); }}
        .badge-rbf {{ background: rgba(244,114,182,0.2); color: #f472b6; border: 1px solid rgba(244,114,182,0.4); }}
        .badge-empate {{ background: rgba(148,163,184,0.2); color: #94a3b8; border: 1px solid rgba(148,163,184,0.4); }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        @media (max-width: 900px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
        }}
        .chart-card {{
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem;
            overflow: hidden;
            box-sizing: border-box;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .chart-card .plotly-graph-div {{
            width: 100% !important;
            margin: 0 auto;
        }}
        .analysis-box {{
            background: rgba(167,139,250,0.06);
            border: 1px solid rgba(167,139,250,0.25);
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 2rem;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .analysis-box h3 {{ color: #a78bfa; margin-bottom: 1rem; }}
        .concl-alert {{
            background: rgba(52,211,153,0.12);
            border-left: 4px solid #34d399;
            padding: 1rem;
            border-radius: 8px;
            color: #e2e8f0;
        }}
        .btn-print {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #a78bfa;
            color: #0b1020;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(167,139,250,0.4);
            z-index: 9999;
        }}
        .btn-print:hover {{ background: #c4b5fd; }}
        @media print {{
            .btn-print {{ display: none; }}
            @page {{
                size: A4 portrait;
                margin: 1cm;
            }}
            body {{
                background: #0b1020 !important;
                color: #e2e8f0 !important;
                padding: 0 !important;
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }}
            .header, .kpi-card, .chart-card, .analysis-box, .report-table {{
                background: #111827 !important;
                border: 1px solid rgba(255,255,255,0.15) !important;
                color: #e2e8f0 !important;
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }}
            .chart-row {{
                display: flex !important;
                flex-direction: row !important;
                gap: 1rem !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                margin-bottom: 1.2rem !important;
            }}
            .chart-card {{
                flex: 1 1 50% !important;
                width: 50% !important;
                max-width: 50% !important;
                box-sizing: border-box !important;
                page-break-inside: avoid !important;
                break-inside: avoid !important;
                overflow: hidden !important;
            }}
            .analysis-box {{
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }}
            .report-table tr {{
                page-break-inside: avoid !important;
                break-inside: avoid !important;
            }}
        }}
    </style>
</head>
<body>
    <button class="btn-print" onclick="window.print()">🖨️ Imprimir / Guardar PDF</button>

    <div class="header">
        <h1>🎯 Informe Comparativo: SVM Lineal vs SVM RBF</h1>
        <p>Evaluación cuantitativa y cualitativa sobre CS:GO Round Snapshots · Generado el {fecha}</p>
        <p style="margin-top: 0.3rem;">Configuración: <b>{n_train:,}</b> muestras de entrenamiento · <b>{n_test:,}</b> de prueba · <b>{cfg['cv']}-fold CV</b> ({SCORINGS.get(cfg['scoring'], cfg['scoring'])})</p>
    </div>

    <div class="kpi-grid">
        {tarjetas_html}
    </div>

    <h2 class="section-title">📊 Comparación Cuantitativa de Desempeño</h2>
    {tabla_html}

    <h2 class="section-title">📈 Visualizaciones y Gráficas Interactivas</h2>
    {graficos_html}

    {analisis_html}

    <footer style="margin-top: 3rem; text-align: center; color: #64748b; font-size: 0.8rem;">
        Informe generado automáticamente desde la aplicación SVM CS:GO · Modelos entrenados con Scikit-Learn
    </footer>
</body>
</html>"""


# ----------------------------------------------------------------------------
# Carga de datos
# ----------------------------------------------------------------------------
with st.spinner("Cargando dataset CS:GO..."):
    DF, T_CARGA = cargar()
N_TOTAL = len(DF)
COLUMNAS = [c for c in DF.columns if c != "round_winner"]

# ----------------------------------------------------------------------------
# Barra lateral
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuración")

    st.markdown("### 📁 Partición de Datos (Modulable)")
    estrategia = st.radio(
        "Esquema de partición",
        ["80% Train / 20% Test", "20% Train / 80% Test", "Personalizado"],
        horizontal=True,
        help="80/20 es el clásico; 20/80 entrena con ~24.5k muestras y evalúa en ~98k del dataset completo."
    )
    if estrategia == "80% Train / 20% Test":
        test_size = 0.20
    elif estrategia == "20% Train / 80% Test":
        test_size = 0.80
    else:
        test_size = st.slider("Proporción de prueba (test)", 0.05, 0.90, 0.20, 0.05)

    n_muestras = st.slider("Tamaño de la muestra total", 1000, N_TOTAL, N_TOTAL, 1000,
                           help="Muestra estratificada del CSV. Se usa la MISMA para ambos modelos.")
    semilla = st.number_input("Semilla aleatoria", 0, 9999, 42)

    st.markdown("### 🔁 Validación cruzada")
    cv = st.slider("Particiones (k-fold)", 2, 10, 5)
    scoring = st.selectbox("Métrica de calibración", list(SCORINGS),
                           format_func=lambda k: SCORINGS[k])
    usar_todos_nucleos = st.checkbox("Usar todos los núcleos de CPU", True)

    st.markdown("### 📏 SVM Lineal")
    Cs_lineal = st.multiselect("Valores de C", [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                               [0.001, 0.01, 0.1, 1.0, 10.0], key="c_lin")

    st.markdown("### 🌀 SVM RBF")
    Cs_rbf = st.multiselect("Valores de C ", [0.01, 0.1, 1.0, 10.0, 100.0],
                            [0.1, 1.0, 10.0], key="c_rbf")
    gammas = st.multiselect("Valores de γ (gamma)", ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                            ["scale", 0.01, 0.1])

    n_train = int(n_muestras * (1 - test_size))
    acelerar_rbf = st.checkbox(
        "⚡ Calibrar RBF en submuestra (Rápido)",
        value=True,
        help="Busca el mejor C y gamma con una submuestra en segundos y luego re-entrena con TODO el entrenamiento."
    )
    if acelerar_rbf and n_train > 1000:
        n_sub_rbf = st.slider("Submuestra para GridSearch (RBF)", 500, min(5000, n_train), min(2000, n_train), 500)
    else:
        n_sub_rbf = None

    st.caption(f"Entrenamiento: **{n_train:,}** ({int((1-test_size)*100)}%) · "
               f"Prueba: **{n_muestras - n_train:,}** ({int(test_size*100)}%)  \n"
               f"Ajustes lineal: **{len(Cs_lineal) * cv + 1}** · "
               f"Ajustes RBF: **{len(Cs_rbf) * len(gammas) * cv + 1}**")

    st.markdown("---")
    c1, c2 = st.columns(2)
    btn_lin = c1.button("▶ Lineal", type="secondary")
    btn_rbf = c2.button("▶ RBF", type="secondary")
    btn_ambos = st.button("⚡ Entrenar ambos", type="primary")
    if st.button("🗑️ Limpiar resultados"):
        st.session_state.resultados.clear()
        st.session_state.fronteras.clear()
        st.rerun()

    if any(k in RES for k in ("lineal", "rbf")):
        st.markdown("---")
        st.markdown("### 📄 Exportar")
        st.download_button(
            label="📥 Descargar Informe HTML",
            data=generar_informe_html(),
            file_name=f"informe_svm_csgo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key="dl_btn_sidebar"
        )

CONFIG_DATOS = dict(n_muestras=n_muestras, test_size=test_size, semilla=int(semilla))

# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🎯 SVM Lineal vs SVM RBF</h1>
  <p>Predicción del ganador de la ronda en <b>CS:GO</b> (CT = +1 · T = −1) ·
  {N_TOTAL:,} rondas · {len(COLUMNAS)} variables · dataset cargado en {fmt_t(T_CARGA)}</p>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------
def ejecutar(tipo):
    Cs = Cs_lineal if tipo == "lineal" else Cs_rbf
    if not Cs or (tipo == "rbf" and not gammas):
        st.error(f"Selecciona al menos un valor de C{' y de γ' if tipo == 'rbf' else ''} "
                 f"para {NOMBRES[tipo]}.")
        return
    (X_train, X_test, y_train, y_test), t_prep = preparar(n_muestras, test_size, int(semilla))
    kw = dict(Cs=Cs, cv=cv, scoring=scoring, n_jobs=-1 if usar_todos_nucleos else 1,
              random_state=int(semilla))
    if tipo == "rbf":
        kw["gammas"] = gammas
        if acelerar_rbf and n_sub_rbf:
            kw["n_submuestra_busqueda"] = n_sub_rbf

    with st.status(f"Entrenando {NOMBRES[tipo]} con {len(y_train):,} muestras...",
                   expanded=True) as estado:
        if tipo == "rbf" and acelerar_rbf and n_sub_rbf and len(y_train) > n_sub_rbf:
            st.write(f"⚡ Calibrando hiperparámetros en submuestra de {n_sub_rbf:,} muestras...")
            busqueda, t_grid = sm.entrenar_modelo(tipo, X_train, y_train, **kw)
            st.write(f"🚀 Re-entrenamiento con {len(y_train):,} muestras completado en {fmt_t(float(busqueda.refit_time_))}.")
        else:
            st.write("🔍 Calibrando hiperparámetros con GridSearchCV...")
            busqueda, t_grid = sm.entrenar_modelo(tipo, X_train, y_train, **kw)

        st.write(f"📊 Evaluando en {len(y_test):,} muestras de prueba (cálculo de inferencia único)...")
        ev = sm.evaluar_modelo(busqueda, X_test, y_test)
        cvres = sm.resumen_cv(busqueda)
        mejor = cvres.loc[cvres["rank_test_score"].idxmin()]
        t_refit_real = float(busqueda.refit_time_)
        estado.update(label=f"✅ {NOMBRES[tipo]} listo (Búsqueda: {fmt_t(t_grid)} · Refit: {fmt_t(t_refit_real)} · Test: {fmt_t(ev['t_pred'])})",
                      state="complete", expanded=False)

    RES[tipo] = {
        "busqueda": busqueda, "eval": ev, "cv": cvres, "mejores": busqueda.best_params_,
        "config": {**CONFIG_DATOS, "cv": cv, "scoring": scoring},
        "n_train": len(y_train), "n_test": len(y_test),
        "n_busqueda": getattr(busqueda, "n_submuestra_", len(y_train)),
        "gap_train_val": float(mejor["mean_train_score"] - mejor["mean_test_score"]),
        "tiempos": {
            "preparacion": t_prep, "gridsearch": t_grid,
            "fit_medio": float(mejor["mean_fit_time"]), "refit": t_refit_real,
            "prediccion": ev["t_pred"], "pred_por_muestra": ev["t_pred"] / len(y_test),
            "total": t_prep + t_grid + t_refit_real + ev["t_pred"],
        },
    }
    st.session_state.fronteras.pop(tipo, None)


if btn_lin or btn_ambos:
    ejecutar("lineal")
if btn_rbf or btn_ambos:
    ejecutar("rbf")


# ----------------------------------------------------------------------------
# Vistas
# ----------------------------------------------------------------------------
def vista_modelo(tipo):
    r = RES.get(tipo)
    if r is None:
        st.info(f"Aún no se ha entrenado el **{NOMBRES[tipo]}**. Configura los parámetros en la "
                f"barra lateral y pulsa **▶ {'Lineal' if tipo == 'lineal' else 'RBF'}**.")
        return
    ev, m, t, c = r["eval"], r["eval"]["metricas"], r["tiempos"], COLORES[tipo]

    badges = "".join(f'<span class="badge" style="color:{c}">{k.replace("svm__", "")} = {v}</span>'
                     for k, v in r["mejores"].items())
    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        st.markdown(f"**Mejores hiperparámetros:** {badges} "
                    f'<span class="badge">entrenamiento: {r["n_train"]:,}</span>'
                    f'<span class="badge">prueba: {r["n_test"]:,}</span>'
                    f'<span class="badge">CV: {r["config"]["cv"]}-fold · {SCORINGS[r["config"]["scoring"]]}</span>',
                    unsafe_allow_html=True)
    with col_b2:
        st.download_button(
            label="📥 Descargar Informe HTML",
            data=generar_informe_html(),
            file_name=f"informe_svm_{tipo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key=f"dl_btn_{tipo}"
        )

    fila_tarjetas([
        ("Exactitud", f"{m['Exactitud'] * 100:.2f}%", f"Error: {m['Tasa de error'] * 100:.2f}%", c),
        ("Precisión", f"{m['Precisión']:.3f}", "VP / (VP + FP)", c),
        ("Recall", f"{m['Sensibilidad (Recall)']:.3f}", "VP / (VP + FN)", c),
        ("F1", f"{m['F1']:.3f}", "media armónica", c),
        ("AUC-ROC", f"{m['AUC-ROC']:.3f}", f"AP: {m['Precisión promedio (AP)']:.3f}", c),
        ("Tiempo total", fmt_t(t["total"]), f"GridSearch: {fmt_t(t['gridsearch'])}", c),
    ])
    fila_tarjetas([
        ("Especificidad", f"{m['Especificidad']:.3f}", "VN / (VN + FP)", c),
        ("Exactitud balanceada", f"{m['Exactitud balanceada']:.3f}", "promedio de recalls", c),
        ("MCC", f"{m['MCC']:.3f}", "corr. de Matthews (−1 a 1)", c),
        ("Re-entrenamiento", fmt_t(t["refit"]), "modelo final", c),
        ("Predicción", fmt_t(t["prediccion"]), f"{t['pred_por_muestra'] * 1e6:.1f} µs / ronda", c),
        ("Zona de margen", f"{ev['extras']['frac_zona_margen'] * 100:.1f}%", "|f(x)| < 1 en test", c),
    ])

    a, b = st.columns(2)
    with a:
        grafico(fig_confusion(tipo), f"cm_{tipo}")
    with b:
        grafico(fig_roc([tipo]), f"roc_{tipo}")
    a, b = st.columns(2)
    with a:
        grafico(fig_pr([tipo]), f"pr_{tipo}")
    with b:
        grafico(fig_validacion(tipo), f"val_{tipo}")
    a, b = st.columns(2)
    with a:
        st.markdown("#### 📋 Reporte de clasificación")
        rep = pd.DataFrame(ev["reporte"]).T
        rep.columns = ["Precisión", "Recall", "F1", "Soporte"]
        st.dataframe(rep.style.format({"Precisión": "{:.3f}", "Recall": "{:.3f}",
                                       "F1": "{:.3f}", "Soporte": "{:,.0f}"}))
        st.markdown("#### 🔎 Resultados de la validación cruzada")
        st.dataframe(r["cv"].sort_values("rank_test_score").style.format(
            {"mean_test_score": "{:.4f}", "std_test_score": "{:.4f}",
             "mean_train_score": "{:.4f}", "mean_fit_time": "{:.3f} s"}), height=230)
    with b:
        grafico(fig_tiempos_modelo(tipo), f"tiempos_{tipo}")
        if tipo == "rbf":
            nsv, (sv_t, sv_ct) = ev["extras"]["n_vectores_soporte"], ev["extras"]["vectores_por_clase"]
            fila_tarjetas([
                ("Vectores de soporte", f"{nsv:,}", f"{nsv / r['n_train'] * 100:.1f}% del entrenamiento", c),
                ("SV clase T", f"{sv_t:,}", "", COLOR_T),
                ("SV clase CT", f"{sv_ct:,}", "", COLOR_CT),
            ])
    if tipo == "lineal":
        grafico(fig_coeficientes(), "coef_lineal")


def vista_comparacion():
    if not all(k in RES for k in ("lineal", "rbf")):
        st.info("Entrena **ambos** modelos para ver la comparación (botón **⚡ Entrenar ambos**).")
        return
    L, R = RES["lineal"], RES["rbf"]
    datos_L = {k: L["config"][k] for k in ("n_muestras", "test_size", "semilla")}
    datos_R = {k: R["config"][k] for k in ("n_muestras", "test_size", "semilla")}
    if datos_L != datos_R:
        st.warning("⚠️ Los modelos se entrenaron con **muestras distintas**. Vuelve a entrenar "
                   "ambos con la misma configuración para una comparación justa.")

    mL, mR = L["eval"]["metricas"], R["eval"]["metricas"]
    tL, tR = L["tiempos"], R["tiempos"]
    ratio = tR["gridsearch"] / max(tL["gridsearch"], 1e-9)
    d_acc = (mR["Exactitud"] - mL["Exactitud"]) * 100

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("### ⚖️ Comparación Cuantitativa y Tiempos")
    with col_h2:
        st.download_button(
            label="📥 Descargar Informe HTML",
            data=generar_informe_html(),
            file_name=f"informe_comparativo_svm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            type="primary",
            key="btn_dl_comp"
        )

    fila_tarjetas([
        ("Exactitud Lineal", f"{mL['Exactitud'] * 100:.2f}%", "", COLORES["lineal"]),
        ("Exactitud RBF", f"{mR['Exactitud'] * 100:.2f}%", "", COLORES["rbf"]),
        ("Diferencia (RBF − Lineal)", f"{d_acc:+.2f} pp", "puntos porcentuales", "#a78bfa"),
        ("Entrenamiento", f"{ratio:.1f}×", "RBF vs Lineal (GridSearch)", "#facc15"),
        ("Predicción", f"{tR['prediccion'] / max(tL['prediccion'], 1e-9):.1f}×",
         "RBF vs Lineal", "#facc15"),
    ])

    filas = []
    for k in mL:
        menor_es_mejor = k == "Tasa de error"
        mejor = "Lineal" if (mL[k] < mR[k]) == menor_es_mejor else "RBF"
        if abs(mL[k] - mR[k]) < 1e-4:
            mejor = "Empate"
        filas.append([k, mL[k], mR[k], mR[k] - mL[k], mejor])
    for nombre, clave in [("Tiempo GridSearch (s)", "gridsearch"), ("Tiempo refit (s)", "refit"),
                          ("Tiempo predicción (s)", "prediccion"), ("Tiempo total (s)", "total")]:
        mejor = "Lineal" if tL[clave] < tR[clave] else "RBF"
        filas.append([nombre, tL[clave], tR[clave], tR[clave] - tL[clave], mejor])
    tabla = pd.DataFrame(filas, columns=["Métrica", "Lineal", "RBF", "Δ (RBF − Lineal)", "Mejor"])

    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("#### 📋 Tabla comparativa")
        st.dataframe(tabla.set_index("Métrica").style.format(
            {"Lineal": "{:.4f}", "RBF": "{:.4f}", "Δ (RBF − Lineal)": "{:+.4f}"}), height=530)
    with b:
        grafico(fig_radar(), "radar")

    a, b = st.columns(2)
    with a:
        grafico(fig_barras_metricas(), "barras_metricas")
    with b:
        grafico(fig_barras_tiempos(), "barras_tiempos")

    a, b = st.columns(2)
    with a:
        grafico(fig_roc(["lineal", "rbf"]), "roc_comp")
    with b:
        grafico(fig_pr(["lineal", "rbf"]), "pr_comp")
    a, b = st.columns(2)
    with a:
        grafico(fig_confusion("lineal"), "cm_comp_lin")
    with b:
        grafico(fig_confusion("rbf"), "cm_comp_rbf")


def vista_cualitativa():
    entrenados = [t for t in ("lineal", "rbf") if t in RES]
    if not entrenados:
        st.info("Entrena al menos un modelo para la evaluación cualitativa.")
        return

    st.markdown("### 🗺️ Fronteras de decisión en 2D")
    st.markdown('<div class="nota">Los modelos reales usan todas las variables; para <b>visualizar</b> '
                'la forma de la frontera se proyectan los datos a 2D y se re-entrena cada SVM con sus '
                'mejores hiperparámetros. Las líneas punteadas blancas marcan f(x) = −1, 0 y +1 '
                '(el margen).</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns([1.2, 1.6, 1, 1])
    modo = a.radio("Proyección", ["PCA (2 componentes)", "2 variables"], horizontal=False)
    variables = None
    if modo == "2 variables":
        defecto = [v for v in ("ct_armor", "t_armor") if v in COLUMNAS] or COLUMNAS[:2]
        variables = b.multiselect("Variables (elige 2)", COLUMNAS, defecto, max_selections=2)
    n_pts = c.slider("Puntos", 300, 4000, 1500, 100)
    gamma_scale = d.checkbox("γ='scale' en 2D (RBF)", True,
                             help="El γ óptimo en ~100 dimensiones suele ser demasiado suave en 2D.")
    if st.button("🎨 Generar fronteras", type="primary"):
        if modo == "2 variables" and (not variables or len(variables) != 2):
            st.error("Selecciona exactamente 2 variables.")
        else:
            for t in entrenados:
                cfg = RES[t]["config"]
                (X_train, _, y_train, _), _ = preparar(cfg["n_muestras"], cfg["test_size"], cfg["semilla"])
                with st.spinner(f"Calculando frontera {NOMBRES[t]}..."):
                    st.session_state.fronteras[t] = sm.frontera_decision_2d(
                        t, RES[t]["mejores"], X_train, y_train,
                        modo="pca" if modo.startswith("PCA") else "vars", variables=variables,
                        n_max=n_pts, gamma_2d_scale=gamma_scale, random_state=cfg["semilla"])
    cols = st.columns(len(entrenados))
    for col, t in zip(cols, entrenados):
        fr = st.session_state.fronteras.get(t)
        with col:
            if fr is None:
                st.caption(f"Pulsa **Generar fronteras** para ver la del {NOMBRES[t]}.")
            else:
                grafico(fig_frontera(t, fr), f"frontera_{t}")
                st.caption(f"Exactitud en 2D: **{fr['acc_2d'] * 100:.1f}%** · ajuste: {fmt_t(fr['t_fit'])} · "
                           f"parámetros: {fr['params']}")

    st.markdown("### 📈 Distribución de la función de decisión")
    st.markdown('<div class="nota">Cuanto más separadas estén las campanas ámbar (T) y azul (CT) '
                'respecto a f(x)=0, más seguro es el clasificador. La masa entre −1 y +1 son rondas '
                'ambiguas dentro del margen.</div>', unsafe_allow_html=True)
    cols = st.columns(len(entrenados))
    for col, t in zip(cols, entrenados):
        with col:
            grafico(fig_distribucion_margen(t), f"dist_{t}")

    st.markdown("### 🧭 Criterios cualitativos")
    criterios = pd.DataFrame({
        "Criterio": ["Forma de la frontera", "Interpretabilidad", "Escalabilidad (n grande)",
                     "Hiperparámetros a calibrar", "Memoria del modelo", "Riesgo de sobreajuste"],
        "SVM Lineal": ["Hiperplano (recta en 2D)", "Alta: pesos w por variable",
                       "Excelente (≈ lineal en n)", "C", f"{len(COLUMNAS)} pesos + sesgo", "Bajo"],
        "SVM RBF": ["Curva flexible, regiones no convexas", "Baja: caja negra basada en similitud",
                    "Limitada (≈ O(n²)–O(n³))", "C y γ",
                    f"{RES['rbf']['eval']['extras']['n_vectores_soporte']:,} vectores de soporte"
                    if "rbf" in RES else "vectores de soporte", "Mayor si γ o C son altos"],
    }).set_index("Criterio")
    st.table(criterios)

    if all(k in RES for k in ("lineal", "rbf")):
        st.markdown("### 🧠 Análisis automático")
        lineas, concl = analisis_automatico()
        st.markdown(lineas)
        st.success(f"**Conclusión:** {concl}")


tab_lin, tab_rbf, tab_comp, tab_cual = st.tabs(
    ["📏 SVM Lineal", "🌀 SVM RBF", "⚖️ Comparación cuantitativa", "🧭 Evaluación cualitativa"])
with tab_lin:
    vista_modelo("lineal")
with tab_rbf:
    vista_modelo("rbf")
with tab_comp:
    vista_comparacion()
with tab_cual:
    vista_cualitativa()
