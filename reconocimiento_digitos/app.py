"""
Interfaz web del sistema de reconocimiento de dígitos (Streamlit).

Ejecución:
    streamlit run app.py

La aplicación permite subir una foto, segmentar uno o varios dígitos,
reconocerlos con el modelo elegido, ver la confianza y consultar las métricas,
matrices de confusión y comparativas generadas durante la evaluación.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from src.dataset import load_dataset_info
from src.prediction import available_models, predict_image
from src.preprocessing import to_uint8
from src.training import load_trained_model, model_exists

# ---------------------------------------------------------------------------
# Configuración de página y estilos
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reconocimiento de Dígitos · ML & DL",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root {
    --primary: #2563eb;
    --primary-dark: #1e3a8a;
    --accent: #10b981;
    --ink: #0f172a;
    --muted: #64748b;
    --surface: #ffffff;
    --line: #e2e8f0;
}
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    font-family: "Segoe UI", "Inter", system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1350px; }
.hero {
    background: radial-gradient(circle at 12% 18%, rgba(37,99,235,.45), transparent 32%),
                radial-gradient(circle at 88% 82%, rgba(16,185,129,.32), transparent 30%),
                linear-gradient(130deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%);
    color: white;
    padding: 2.1rem 2.4rem 1.9rem 2.4rem;
    border-radius: 24px;
    margin-bottom: 1.4rem;
    box-shadow: 0 22px 45px -22px rgba(15, 23, 42, .65);
}
.hero h1 { color: white; font-size: 2.15rem; margin: .35rem 0 .5rem 0; letter-spacing: -.02em; }
.hero p { color: #dbeafe; font-size: 1.02rem; margin: 0; max-width: 850px; line-height: 1.6; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.25);
    padding: .32rem .8rem;
    border-radius: 999px;
    font-size: .76rem;
    font-weight: 600;
    letter-spacing: .09em;
    text-transform: uppercase;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.05rem 1.2rem;
    box-shadow: 0 10px 25px -20px rgba(15,23,42,.45);
    height: 100%;
}
.metric-label { color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.metric-value { color: var(--ink); font-size: 1.65rem; font-weight: 800; margin-top: .25rem; letter-spacing: -.02em; }
.metric-hint { color: var(--muted); font-size: .78rem; margin-top: .25rem; }
.result-card {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
    color: white;
    border-radius: 22px;
    padding: 1.45rem 1.6rem 1.3rem 1.6rem;
    box-shadow: 0 18px 35px -20px rgba(29,78,216,.8);
}
.result-label { color: #bfdbfe; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
.result-number { font-size: 3.5rem; font-weight: 900; line-height: 1.05; margin: .25rem 0 .1rem 0; letter-spacing: .06em; }
.result-confidence { color: #d1fae5; font-size: .98rem; font-weight: 600; }
.digit-row {
    display: flex; align-items: center; gap: .8rem;
    background: white; border: 1px solid var(--line); border-radius: 14px;
    padding: .55rem .75rem; margin-bottom: .5rem;
}
.digit-chip {
    width: 43px; height: 43px; border-radius: 11px; background: #eff6ff;
    color: #1d4ed8; display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1.35rem; flex-shrink: 0;
}
.digit-bar-track { background: #e2e8f0; border-radius: 999px; height: 9px; width: 100%; overflow: hidden; }
.digit-bar-fill { background: linear-gradient(90deg, #3b82f6, #10b981); height: 9px; border-radius: 999px; }
.small-muted { color: var(--muted); font-size: .8rem; }
.pill {
    display: inline-block; padding: .2rem .55rem; border-radius: 999px;
    background: #dcfce7; color: #166534; font-size: .74rem; font-weight: 700;
}
.pill-warn { background: #fef3c7; color: #92400e; }
.pill-error { background: #fee2e2; color: #991b1b; }
.section-title { color: var(--ink); font-weight: 800; font-size: 1.18rem; margin: .2rem 0 .7rem 0; }
div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
div[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; }
.stTabs [data-baseweb="tab"] {
    background: white; border: 1px solid var(--line); border-radius: 11px;
    padding: .55rem 1rem; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #eff6ff; border-color: #bfdbfe; color: #1d4ed8; }
.stButton > button, .stDownloadButton > button { border-radius: 11px; font-weight: 600; }
code { color: #1e3a8a; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Carga de artefactos con caché
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(name: str) -> Any:
    """Carga y memoriza un modelo entrenado."""
    return load_trained_model(name)


@st.cache_data(show_spinner=False)
def get_metrics() -> pd.DataFrame:
    """Carga la tabla de métricas de test."""
    if config.METRICS_FILE.exists():
        return pd.read_csv(config.METRICS_FILE)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_json(path_string: str) -> dict[str, Any]:
    """Carga un JSON cacheado."""
    path = Path(path_string)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def get_demo_manifest() -> list[dict[str, Any]]:
    """Devuelve las imágenes de ejemplo disponibles."""
    path = config.SAMPLES_DIR / "demo_manifest.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def read_text(path_string: str) -> str:
    path = Path(path_string)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def metric_card(label: str, value: str, hint: str = "") -> str:
    """HTML de una tarjeta de métrica."""
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-hint">{hint}</div></div>'
    )


def confidence_class(value: float) -> str:
    if value >= 0.90:
        return "pill"
    if value >= 0.70:
        return "pill pill-warn"
    return "pill pill-error"


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Encabezado y barra lateral
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-badge">Machine Learning · Deep Learning · Visión por computadora</div>
  <h1>Reconocimiento de Dígitos Manuscritos</h1>
  <p>Sistema académico que entrena y compara Regresión Logística, K-Means, KNN, SVM,
  una red neuronal densa y una CNN sobre MNIST. Sube una imagen con uno o varios
  números: el sistema los segmenta, los clasifica y muestra la confianza de cada predicción.</p>
</div>
""",
    unsafe_allow_html=True,
)

models_status = available_models()
trained_models = [name for name, exists in models_status.items() if exists]

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    if trained_models:
        default_index = (
            trained_models.index("cnn") if "cnn" in trained_models else 0
        )
        model_name = st.selectbox(
            "Modelo de reconocimiento",
            options=trained_models,
            index=default_index,
            format_func=lambda name: config.MODEL_DISPLAY_NAMES[name],
        )
        st.caption(
            "La CNN suele ser el modelo más preciso. La MLP y los modelos "
            "clásicos permiten comparar enfoques."
        )
    else:
        model_name = None
        st.error(
            "No hay modelos entrenados. Ejecuta en la terminal:\n\n"
            "`python main.py train --model all`"
        )

    st.divider()
    st.markdown("### 🎛️ Parámetros de segmentación")
    min_area = st.slider(
        "Área mínima de un dígito (px²)",
        min_value=10,
        max_value=400,
        value=config.DEFAULT_MIN_AREA,
        step=10,
        help="Súbelo si la imagen tiene mucho ruido; bájalo para dígitos pequeños.",
    )
    max_digits = st.slider(
        "Máximo de dígitos a detectar",
        min_value=1,
        max_value=15,
        value=config.DEFAULT_MAX_DIGITS,
    )

    st.divider()
    metrics_sidebar = get_metrics()
    if model_name and not metrics_sidebar.empty:
        row = metrics_sidebar[metrics_sidebar["modelo"] == model_name]
        if not row.empty:
            st.markdown("### 📈 Rendimiento en test")
            st.metric("Accuracy", f"{row.iloc[0]['accuracy'] * 100:.2f} %")
            st.metric("F1-score macro", f"{row.iloc[0]['f1_macro'] * 100:.2f} %")
            st.caption(f"Evaluado sobre {int(row.iloc[0]['n_test']):,} imágenes de test.")

    st.divider()
    st.markdown("### 📚 Dataset")
    st.markdown(
        "<div class='small-muted'>MNIST · 70 000 imágenes de 28×28 px<br>"
        "54 000 entrenamiento · 6 000 validación · 10 000 test</div>",
        unsafe_allow_html=True,
    )
    st.caption("Las métricas finales se calculan solo sobre el test oficial.")

    st.divider()
    st.markdown(
        "<div class='small-muted'>Proyecto académico de Machine Learning y "
        "Deep Learning. Los resultados mostrados proceden de ejecuciones reales."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pestaña 1 · Reconocimiento
# ---------------------------------------------------------------------------
def render_recognition() -> None:
    st.markdown('<div class="section-title">Reconoce un número escrito a mano</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1], gap="large")

    manifest = get_demo_manifest()
    sample_options = {item["file"]: item for item in manifest}

    with left:
        source_mode = st.radio(
            "Fuente de la imagen",
            options=["Subir imagen", "Usar ejemplo"],
            horizontal=True,
            label_visibility="collapsed",
        )

        image_source: Any = None
        expected: str | None = None

        if source_mode == "Subir imagen":
            uploaded = st.file_uploader(
                "Carga una foto o captura con uno o varios dígitos",
                type=["png", "jpg", "jpeg", "bmp", "webp"],
                help="Funciona mejor con trazos oscuros sobre fondo claro y buena iluminación.",
            )
            if uploaded is not None:
                image_source = uploaded.getvalue()
        else:
            if sample_options:
                selected_file = st.selectbox(
                    "Imagen de ejemplo",
                    options=list(sample_options.keys()),
                    format_func=lambda name: sample_options[name]["description"],
                )
                image_source = config.SAMPLES_DIR / selected_file
                expected = sample_options[selected_file]["expected"]
                st.caption(
                    "Ejemplos generados a partir de imágenes del test de MNIST "
                    "con rotación, ruido y desenfoque. No son fotos reales."
                )
            else:
                st.info(
                    "Todavía no hay ejemplos. Entrena y ejecuta "
                    "`python main.py demo` para generarlos."
                )

        if image_source is not None and model_name is not None:
            try:
                with st.spinner("Procesando imagen y reconociendo dígitos..."):
                    model = get_model(model_name)
                    prediction = predict_image(
                        image_source,
                        model,
                        model_name,
                        min_area=min_area,
                        max_digits=max_digits,
                    )

                if prediction.digits:
                    conf = prediction.mean_confidence
                    st.markdown(
                        f"""
<div class="result-card">
  <div class="result-label">Número reconocido</div>
  <div class="result-number">{prediction.number}</div>
  <div class="result-confidence">Confianza media: {conf * 100:.1f} % ·
  {len(prediction.digits)} dígito(s) detectado(s)</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    if expected is not None:
                        if prediction.number == expected:
                            st.success(f"Coincide con el valor esperado: {expected}")
                        else:
                            st.warning(
                                f"El valor esperado era {expected} y el modelo "
                                f"reconoció {prediction.number}."
                            )

                    if prediction.fallback:
                        st.info(
                            "No se detectaron componentes separados; se procesó la "
                            "imagen completa como un único dígito."
                        )

                    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
                    for item in prediction.predictions:
                        color = "#10b981" if item.confidence >= 0.9 else (
                            "#f59e0b" if item.confidence >= 0.7 else "#ef4444"
                        )
                        st.markdown(
                            f"""
<div class="digit-row">
  <div class="digit-chip">{item.digit}</div>
  <div style="flex:1">
    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
      <span style="font-weight:600; color:#0f172a;">Dígito {item.index + 1}</span>
      <span style="font-weight:700; color:{color};">{item.confidence * 100:.1f} %</span>
    </div>
    <div class="digit-bar-track"><div class="digit-bar-fill" style="width:{min(100, item.confidence * 100):.1f}%; background:{color};"></div></div>
  </div>
</div>
""",
                            unsafe_allow_html=True,
                        )

                    with st.expander("Ver probabilidades por clase del primer dígito"):
                        probs = prediction.predictions[0].probabilities
                        probs_df = pd.DataFrame(
                            {
                                "Dígito": [str(i) for i in range(10)],
                                "Probabilidad (%)": [round(p * 100, 2) for p in probs],
                            }
                        )
                        st.bar_chart(
                            probs_df,
                            x="Dígito",
                            y="Probabilidad (%)",
                            color="#2563eb",
                            height=260,
                        )

                    annotated_rgb = bgr_to_rgb(prediction.annotated_image)
                    success, buffer = cv2.imencode(".png", prediction.annotated_image)
                    if success:
                        st.download_button(
                            "⬇️ Descargar imagen anotada",
                            data=buffer.tobytes(),
                            file_name="prediccion_digitos.png",
                            mime="image/png",
                            width="stretch",
                        )
                else:
                    st.error("No se pudo reconocer ningún dígito en la imagen.")
                    annotated_rgb = bgr_to_rgb(prediction.annotated_image)

            except Exception as exc:
                st.error(f"No se pudo procesar la imagen: {exc}")
                annotated_rgb = None
        else:
            annotated_rgb = None
            if image_source is None and model_name is not None:
                st.info(
                    "Sube una imagen o elige un ejemplo para comenzar. "
                    "También puedes probar desde la terminal: "
                    "`python main.py predict --image foto.png --model cnn`"
                )

    with right:
        if image_source is not None and model_name is not None and annotated_rgb is not None:
            st.markdown("#### Imagen original")
            st.image(bgr_to_rgb(prediction.original_image), width="stretch")

            st.markdown("#### Preprocesamiento")
            preview_cols = st.columns(2)
            with preview_cols[0]:
                st.image(
                    prediction.binary_image,
                    caption="Umbralización de Otsu",
                    width="stretch",
                    clamp=True,
                )
            with preview_cols[1]:
                st.image(
                    annotated_rgb,
                    caption="Segmentación y predicción",
                    width="stretch",
                )

            if prediction.predictions:
                st.markdown("#### Dígitos preparados para el modelo (28×28)")
                n_cols = min(len(prediction.predictions), 5)
                cols = st.columns(n_cols)
                for i, item in enumerate(prediction.predictions):
                    with cols[i % n_cols]:
                        st.image(
                            to_uint8(item.prepared),
                            caption=f"{item.digit} · {item.confidence * 100:.0f} %",
                            width="stretch",
                            clamp=True,
                        )
        else:
            st.markdown("#### Vista previa")
            st.info(
                "Aquí verás la imagen original, la máscara binarizada, los recortes "
                "de cada dígito y la predicción con su nivel de confianza."
            )
            if config.PLOTS_DIR.joinpath("sample_predictions_cnn.png").exists():
                st.image(
                    str(config.PLOTS_DIR / "sample_predictions_cnn.png"),
                    caption="Ejemplos de predicción sobre el conjunto de test",
                    width="stretch",
                )

    st.divider()
    st.markdown("#### ¿Cómo interpretar la confianza?")
    st.markdown(
        """
- **≥ 90 %**: predicción muy segura; el trazo se parece claramente al dígito.
- **70–90 %**: predicción probable; conviene revisar el recorte y la máscara.
- **< 70 %**: el modelo duda. Puede deberse a ruido, un trazo atípico, dígitos
  pegados o un encuadre pobre. Ajusta el área mínima en la barra lateral.

En el SVM la "confianza" se obtiene aplicando softmax a los márgenes de
decisión, por lo que es una medida relativa y no una probabilidad calibrada.
"""
    )


# ---------------------------------------------------------------------------
# Pestaña 2 · Evaluación
# ---------------------------------------------------------------------------
def render_evaluation() -> None:
    metrics = get_metrics()
    st.markdown('<div class="section-title">Evaluación sobre el conjunto de test</div>', unsafe_allow_html=True)

    if metrics.empty:
        st.warning(
            "Aún no hay métricas. Entrena los modelos y ejecuta la evaluación:\n\n"
            "```bash\npython main.py train --model all\npython main.py evaluate\n```"
        )
        return

    best = metrics.sort_values("accuracy", ascending=False).iloc[0]
    cols = st.columns(4)
    cards = [
        ("Mejor modelo", best["nombre"], f"Accuracy {best['accuracy'] * 100:.2f} %"),
        ("Modelos evaluados", str(len(metrics)), "Clasificadores supervisados"),
        ("Mejor F1 macro", f"{metrics['f1_macro'].max() * 100:.2f} %", "Media no ponderada"),
        ("Imágenes de test", f"{int(best['n_test']):,}", "Test oficial de MNIST"),
    ]
    for column, (label, value, hint) in zip(cols, cards):
        with column:
            st.markdown(metric_card(label, value, hint), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    table = metrics.sort_values("accuracy", ascending=False).copy()
    display = pd.DataFrame(
        {
            "Modelo": table["nombre"],
            "Accuracy (%)": (table["accuracy"] * 100).round(2),
            "Precision macro (%)": (table["precision_macro"] * 100).round(2),
            "Recall macro (%)": (table["recall_macro"] * 100).round(2),
            "F1 macro (%)": (table["f1_macro"] * 100).round(2),
            "F1 ponderado (%)": (table["f1_weighted"] * 100).round(2),
            "Inferencia (s)": table["tiempo_inferencia_s"].round(3),
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Matriz de confusión y métricas por clase")
    model_options = {row["nombre"]: row["modelo"] for _, row in table.iterrows()}
    selected_display = st.selectbox("Modelo a inspeccionar", list(model_options.keys()))
    selected = model_options[selected_display]

    col_cm, col_class = st.columns([1.35, 1], gap="large")
    with col_cm:
        cm_path = config.PLOTS_DIR / f"confusion_matrix_{selected}.png"
        if cm_path.exists():
            st.image(str(cm_path), width="stretch")
        else:
            st.info("La matriz de confusión se genera al ejecutar `python main.py evaluate`.")

    with col_class:
        csv_path = config.RESULTS_DIR / f"classification_{selected}.csv"
        if csv_path.exists():
            per_class = pd.read_csv(csv_path)
            per_class.columns = ["Dígito", "Precision", "Recall", "F1", "Soporte"]
            for column in ("Precision", "Recall", "F1"):
                per_class[column] = (per_class[column] * 100).round(2)
            st.dataframe(per_class, width="stretch", hide_index=True, height=390)
        report_path = config.RESULTS_DIR / f"classification_report_{selected}.txt"
        if report_path.exists():
            with st.expander("Ver informe completo"):
                st.code(read_text(str(report_path)), language="text")

    st.divider()
    st.markdown("#### Curvas de entrenamiento")
    curve_models = [name for name in ("cnn", "mlp") if (config.PLOTS_DIR / f"training_{name}.png").exists()]
    if curve_models:
        curve_name = st.selectbox(
            "Red neuronal",
            options=curve_models,
            format_func=lambda name: config.MODEL_DISPLAY_NAMES[name],
        )
        st.image(str(config.PLOTS_DIR / f"training_{curve_name}.png"), width="stretch")
        st.caption(
            "La validación se usa para early stopping; el test no participa en el "
            "entrenamiento ni en la selección de la mejor época."
        )
    else:
        st.info("Entrena la MLP o la CNN para ver las curvas de precisión y pérdida.")

    st.divider()
    st.markdown("#### Validación cruzada de modelos tradicionales")
    if config.CV_RESULTS_FILE.exists():
        cv = pd.read_csv(config.CV_RESULTS_FILE)
        view = pd.DataFrame(
            {
                "Modelo": cv["nombre"],
                "Folds": cv["folds"],
                "Muestras": cv["n_muestras"],
                "Accuracy media (%)": (cv["accuracy_mean"] * 100).round(2),
                "Accuracy std (%)": (cv["accuracy_std"] * 100).round(2),
                "F1 macro media (%)": (cv["f1_macro_mean"] * 100).round(2),
                "Tiempo (s)": cv["tiempo_total_s"].round(1),
            }
        )
        st.dataframe(view, width="stretch", hide_index=True)
        st.caption(
            "Validación cruzada estratificada de 3 folds sobre un subconjunto de "
            "6 000 imágenes, por viabilidad computacional. Las métricas finales "
            "siempre se reportan sobre test."
        )
    else:
        st.info("Ejecuta `python main.py evaluate` para generar la validación cruzada.")

    sample_path = config.PLOTS_DIR / f"sample_predictions_{selected}.png"
    if sample_path.exists():
        st.divider()
        st.markdown("#### Ejemplos reales de test con la predicción del modelo")
        st.image(str(sample_path), width="stretch")


# ---------------------------------------------------------------------------
# Pestaña 3 · Comparación
# ---------------------------------------------------------------------------
def render_comparison() -> None:
    st.markdown('<div class="section-title">Comparación de algoritmos</div>', unsafe_allow_html=True)
    comparison_path = config.PLOTS_DIR / "model_comparison.png"
    metrics = get_metrics()

    if comparison_path.exists():
        st.image(str(comparison_path), width="stretch")
    elif metrics.empty:
        st.warning("Ejecuta la evaluación para generar la comparación.")
        return

    if not metrics.empty:
        table = metrics.sort_values("accuracy", ascending=False)
        view = pd.DataFrame(
            {
                "Modelo": table["nombre"],
                "Accuracy (%)": (table["accuracy"] * 100).round(2),
                "Precision (%)": (table["precision_macro"] * 100).round(2),
                "Recall (%)": (table["recall_macro"] * 100).round(2),
                "F1 macro (%)": (table["f1_macro"] * 100).round(2),
            }
        )
        st.dataframe(view, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Diferencias conceptuales entre los algoritmos")
    with st.expander("Regresión Logística · modelo lineal supervisado", expanded=False):
        st.markdown(
            "Aprende un peso por píxel y clase, y aplica softmax para obtener "
            "probabilidades. Es rápido e interpretable, pero solo separa clases "
            "con fronteras lineales; por eso rinde menos que KNN, SVM o una CNN."
        )
    with st.expander("KNN · clasificación supervisada basada en instancias", expanded=False):
        st.markdown(
            "No aprende parámetros: guarda el conjunto de entrenamiento y asigna a "
            "cada imagen la clase mayoritaria entre sus *k* vecinos más cercanos. "
            "Con 784 píxeles sufre la maldición de la dimensionalidad, así que se "
            "aplica PCA(50) antes de calcular distancias."
        )
    with st.expander("SVM · clasificación supervisada con márgenes máximos", expanded=False):
        st.markdown(
            "Busca el hiperplano que maximiza el margen entre clases. Con kernel RBF "
            "puede aprender fronteras no lineales. Se entrena con un subconjunto "
            "estratificado de 12 000 imágenes por coste computacional; la evaluación "
            "se hace sobre las 10 000 de test."
        )
    with st.expander("MLP · red neuronal artificial supervisada", expanded=False):
        st.markdown(
            "Dos capas densas con ReLU y dropout aprenden combinaciones no lineales "
            "de píxeles. No aprovecha la estructura espacial de la imagen: al "
            "aplanarla se pierde la noción de vecindad entre píxeles."
        )
    with st.expander("CNN · red neuronal convolucional supervisada", expanded=False):
        st.markdown(
            "Las convoluciones comparten pesos y detectan bordes, curvas y patrones "
            "locales; MaxPooling aporta invariancia a pequeños desplazamientos. Es la "
            "arquitectura más adecuada para imágenes y normalmente la de mayor accuracy."
        )
    with st.expander("K-Means · agrupamiento NO supervisado", expanded=False):
        st.markdown(
            "K-Means **no es un clasificador**: agrupa las imágenes por similitud sin "
            "usar etiquetas. Sus métricas correctas son ARI, NMI, silueta y pureza. "
            "Asignar después un dígito a cada clúster es solo una interpretación "
            "a posteriori, no un aprendizaje supervisado. En esta interfaz se muestra "
            "en una pestaña separada para no compararlo injustamente por accuracy."
        )

    st.info(
        "Nota metodológica: K-Means no aparece en la tabla comparativa de "
        "clasificadores porque su tarea es distinta. Mezclar su 'accuracy tras "
        "mapeo' con la de modelos supervisados sería engañoso."
    )


# ---------------------------------------------------------------------------
# Pestaña 4 · K-Means
# ---------------------------------------------------------------------------
def render_kmeans() -> None:
    st.markdown('<div class="section-title">K-Means: agrupamiento no supervisado</div>', unsafe_allow_html=True)
    km = get_json(str(config.KMEANS_METRICS_FILE))

    if not km:
        st.warning(
            "K-Means aún no está evaluado. Ejecuta:\n\n"
            "```bash\npython main.py train --model kmeans\npython main.py evaluate\n```"
        )
        return

    st.markdown(
        "K-Means recibió **solo píxeles**, nunca etiquetas. Buscó 10 grupos por "
        "similitud. Las métricas siguientes evalúan la calidad del agrupamiento; "
        "la asignación de un dígito a cada clúster se hace después para interpretarlo."
    )

    cols = st.columns(4)
    cards = [
        ("ARI", f"{km['adjusted_rand_index']:.4f}", "Rand ajustado vs. etiquetas"),
        ("NMI", f"{km['normalized_mutual_info']:.4f}", "Información mutua normalizada"),
        ("Silueta", f"{km['silhouette']:.4f}", "Cohesión y separación"),
        ("Pureza", f"{km['purity'] * 100:.2f} %", "Mayoría por clúster"),
    ]
    for column, (label, value, hint) in zip(cols, cards):
        with column:
            st.markdown(metric_card(label, value, hint), unsafe_allow_html=True)

    st.caption(
        f"Exactitud tras mapear cada clúster a su dígito mayoritario: "
        f"{km['exactitud_tras_mapeo'] * 100:.2f} % · Inercia: {km['inercia']:.1f}. "
        "Esta cifra es interpretativa, no una métrica de clasificación supervisada."
    )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        path = config.PLOTS_DIR / "kmeans_centers.png"
        if path.exists():
            st.image(str(path), width="stretch")
    with col_b:
        path = config.PLOTS_DIR / "kmeans_cluster_sizes.png"
        if path.exists():
            st.image(str(path), width="stretch")

    path = config.PLOTS_DIR / "kmeans_clusters_pca.png"
    if path.exists():
        st.markdown("#### Proyección PCA: clústeres vs. etiquetas reales")
        st.image(str(path), width="stretch")

    col_c, col_d = st.columns([1, 1])
    with col_c:
        path = config.PLOTS_DIR / "kmeans_confusion.png"
        if path.exists():
            st.markdown("#### Clústeres mapeados a dígitos")
            st.image(str(path), width="stretch")
    with col_d:
        mapping = km.get("mapeo_cluster_a_digito", {})
        mapping_df = pd.DataFrame(
            {
                "Clúster": list(mapping.keys()),
                "Dígito mayoritario": list(mapping.values()),
            }
        )
        st.markdown("#### Mapeo aprendido (a posteriori)")
        st.dataframe(mapping_df, width="stretch", hide_index=True)
        st.markdown(
            "Si dos clústeres comparten el mismo dígito mayoritario, K-Means dividió "
            "una clase en subgrupos por variaciones de estilo (grosor, inclinación). "
            "Esto es habitual y muestra una limitación del agrupamiento rígido."
        )


# ---------------------------------------------------------------------------
# Pestaña 5 · Guía
# ---------------------------------------------------------------------------
def render_guide() -> None:
    st.markdown('<div class="section-title">Guía de uso y documentación</div>', unsafe_allow_html=True)

    info = {}
    if config.DATASET_INFO_FILE.exists():
        info = load_dataset_info()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 1 · Flujo de la aplicación")
        st.markdown(
            """
1. Elige el modelo en la barra lateral.
2. Sube una foto o selecciona un ejemplo.
3. OpenCV convierte a grises, mejora el contraste, elimina ruido y umbraliza.
4. Se buscan contornos, se filtran y se ordenan de izquierda a derecha.
5. Cada recorte se escala a 20×20, se centra por masa y se normaliza en 28×28.
6. El modelo devuelve una probabilidad por dígito.
7. Se muestra el número completo, la confianza y las imágenes de apoyo.
"""
        )

        st.markdown("#### 2 · Comandos útiles")
        st.code(
            "python main.py prepare\n"
            "python main.py train --model all\n"
            "python main.py evaluate\n"
            "python main.py demo\n"
            "python main.py predict --image foto.png --model cnn\n"
            "python main.py info\n"
            "streamlit run app.py",
            language="bash",
        )

    with col_b:
        st.markdown("#### 3 · Dataset")
        if info:
            st.markdown(
                f"""
- **Nombre:** {info['nombre']}
- **Entrenamiento:** {info['conteos']['train']:,} imágenes
- **Validación:** {info['conteos']['validation']:,} imágenes
- **Test:** {info['conteos']['test']:,} imágenes
- **Dimensiones:** {info['dimensiones_imagen'][0]}×{info['dimensiones_imagen'][1]} px, 1 canal
- **Clases:** 0–9
- **División:** {info['estrategia_division']}
"""
            )
        else:
            st.info("Ejecuta `python main.py prepare` para descargar MNIST.")

        st.markdown("#### 4 · Recomendaciones para fotos reales")
        st.markdown(
            """
- Usa fondo claro y trazo oscuro, con buena iluminación.
- Recorta la zona del número si la imagen es muy grande.
- Deja espacio entre dígitos para facilitar la segmentación.
- Si se detecta ruido, sube el área mínima en la barra lateral.
- Si no se detecta nada, baja el área mínima o revisa el contraste.
- La app procesa una línea de dígitos; para varias líneas, sube cada línea por separado.
"""
        )

    st.divider()
    st.markdown("#### 5 · Limitaciones declaradas")
    st.markdown(
        """
- El SVM se entrena con 12 000 imágenes (subconjunto estratificado) por coste
  computacional. Se indica en la tabla y en el README.
- El KNN usa PCA(50) para que la distancia euclídea sea tratable.
- K-Means es no supervisado y se evalúa con ARI, NMI, silueta y pureza.
- Las imágenes de ejemplo del apartado "Usar ejemplo" son composiciones
  sintéticas de dígitos del test de MNIST con rotación y ruido; no son fotos
  reales y así se indica en la interfaz.
- El reconocimiento está pensado para dígitos individuales o números en una
  línea. Los dígitos muy pegados pueden requerir ajuste manual o fallar.
- La confianza del SVM es una transformación softmax de sus márgenes, no una
  probabilidad calibrada.
"""
    )

    st.divider()
    st.markdown("#### 6 · Estructura del proyecto")
    st.code(
        "reconocimiento_digitos/\n"
        "├── app.py                 Interfaz Streamlit\n"
        "├── main.py                CLI (prepare/train/evaluate/demo/predict)\n"
        "├── config.py              Rutas e hiperparámetros\n"
        "├── requirements.txt\n"
        "├── README.md\n"
        "├── src/\n"
        "│   ├── dataset.py         Descarga y división de MNIST\n"
        "│   ├── preprocessing.py   Normalización y centrado\n"
        "│   ├── segmentation.py    OpenCV: contornos y recortes\n"
        "│   ├── models.py          Definición de los 6 modelos\n"
        "│   ├── training.py        Entrenamiento y persistencia\n"
        "│   ├── evaluation.py      Métricas y gráficos\n"
        "│   ├── prediction.py      Inferencia sobre imágenes nuevas\n"
        "│   └── demo.py            Muestras de demostración\n"
        "├── models/  results/  plots/  data/  tests/",
        language="text",
    )


# ---------------------------------------------------------------------------
# Pestañas principales
# ---------------------------------------------------------------------------
tabs = st.tabs(
    [
        "🔍 Reconocer",
        "📊 Evaluación",
        "⚖️ Comparación",
        "🧩 K-Means",
        "📚 Guía",
    ]
)

with tabs[0]:
    render_recognition()
with tabs[1]:
    render_evaluation()
with tabs[2]:
    render_comparison()
with tabs[3]:
    render_kmeans()
with tabs[4]:
    render_guide()

st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:.78rem; margin-top:2rem;'>"
    "Sistema Inteligente de Reconocimiento y Clasificación de Dígitos Manuscritos · "
    "MNIST · Machine Learning y Deep Learning</div>",
    unsafe_allow_html=True,
)
