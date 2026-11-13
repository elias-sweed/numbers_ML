"""
Interfaz web del sistema de reconocimiento de dígitos manuscritos (Streamlit).

Ejecución:
    streamlit run app.py

Flujo:
    1. Elegir el modelo de reconocimiento.
    2. Cargar una imagen con uno o varios dígitos.
    3. Pulsar "Reconocer dígitos".
    4. Ver la imagen original, la imagen procesada y el número reconocido.

La aplicación usa el pipeline real del proyecto: OpenCV segmenta los dígitos y
el modelo entrenado los clasifica. No se simulan predicciones.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

import config
from src.prediction import available_models, predict_image
from src.preprocessing import to_uint8
from src.training import load_trained_model

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sistema de Reconocimiento de Dígitos Manuscritos",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 2.5rem; }
h1 { color: #1e3a8a; letter-spacing: -.01em; }
.result-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin: .4rem 0 1rem 0;
}
.result-label { color: #1d4ed8; font-size: .8rem; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; }
.result-number { color: #1e3a8a; font-size: 3.2rem; font-weight: 800;
    letter-spacing: .12em; line-height: 1.1; }
.result-detail { color: #475569; font-size: .95rem; }
.small-note { color: #64748b; font-size: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model(name: str) -> Any:
    """Carga un modelo entrenado y lo reutiliza entre ejecuciones."""
    return load_trained_model(name)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR de OpenCV a RGB para mostrarla en Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Encabezado y descripción
# ---------------------------------------------------------------------------
st.title("Sistema de Reconocimiento de Dígitos Manuscritos")
st.markdown(
    "Este sistema reconoce números escritos a mano en una imagen o en el "
    "recorte de un formulario. Convierte la imagen a escala de grises, separa "
    "los dígitos, los prepara y los clasifica con el modelo que elijas."
)
st.caption(
    "Reconoce una línea de dígitos; no procesa formularios completos con "
    "varias filas ni campos de texto."
)

# ---------------------------------------------------------------------------
# 1 y 2 · Modelo e imagen
# ---------------------------------------------------------------------------
modelos_disponibles = [name for name, exists in available_models().items() if exists]

if not modelos_disponibles:
    st.error(
        "No hay modelos entrenados. Ejecuta en la terminal: "
        "`python main.py train --model all`"
    )
    st.stop()

st.subheader("1. Selecciona el modelo")
modelo = st.selectbox(
    "Modelo de reconocimiento",
    options=modelos_disponibles,
    index=modelos_disponibles.index("cnn") if "cnn" in modelos_disponibles else 0,
    format_func=lambda name: config.MODEL_DISPLAY_NAMES[name],
)
st.caption(
    "La CNN suele ofrecer los mejores resultados. Los demás modelos permiten "
    "comparar enfoques de Machine Learning."
)

st.subheader("2. Carga la imagen")
archivo = st.file_uploader(
    "Imagen con uno o varios dígitos manuscritos",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="Funciona mejor con trazos oscuros sobre fondo claro y buena iluminación.",
)

# Una imagen nueva o un modelo distinto invalidan el resultado anterior.
if archivo is None:
    firma_archivo = None
else:
    firma_archivo = f"{archivo.name}:{archivo.size}"

if st.session_state.get("firma_archivo") != firma_archivo:
    st.session_state["firma_archivo"] = firma_archivo
    st.session_state.pop("resultado", None)

if archivo is not None:
    st.image(
        archivo.getvalue(),
        caption=f"Imagen cargada: {archivo.name}",
        width=320,
    )

st.subheader("3. Reconoce los dígitos")
reconocer = st.button(
    "Reconocer dígitos",
    type="primary",
    disabled=archivo is None,
)

if reconocer and archivo is not None:
    try:
        with st.spinner("Procesando la imagen y reconociendo los dígitos..."):
            modelo_entrenado = get_model(modelo)
            prediccion = predict_image(
                archivo.getvalue(),
                modelo_entrenado,
                modelo,
            )
        st.session_state["resultado"] = prediccion
        st.session_state["resultado_modelo"] = modelo
        st.session_state["resultado_archivo"] = archivo.name
    except Exception as exc:  # mensaje claro para el usuario
        st.session_state.pop("resultado", None)
        st.error(
            "No se pudo procesar la imagen. Comprueba que el archivo sea una "
            f"imagen válida e inténtalo de nuevo. Detalle: {exc}"
        )

# ---------------------------------------------------------------------------
# 4 · Resultado
# ---------------------------------------------------------------------------
resultado = st.session_state.get("resultado")
resultado_vigente = (
    resultado is not None
    and st.session_state.get("resultado_modelo") == modelo
    and st.session_state.get("resultado_archivo") == (archivo.name if archivo else None)
)

if archivo is None:
    st.info("Carga una imagen para comenzar.")
elif not resultado_vigente:
    st.info("Pulsa **Reconocer dígitos** para procesar la imagen.")
else:
    st.subheader("4. Resultado del reconocimiento")

    sin_trazo = float(np.mean(resultado.binary_image > 0)) < 0.001

    if sin_trazo:
        st.error(
            "No se ha detectado ningún trazo en la imagen. Prueba con una "
            "imagen más nítida, con el número más grande o con mejor contraste."
        )
    else:
        st.markdown(
            f"""
<div class="result-box">
  <div class="result-label">Número reconocido</div>
  <div class="result-number">{resultado.number}</div>
  <div class="result-detail">
    Modelo: {config.MODEL_DISPLAY_NAMES[modelo]} ·
    {len(resultado.digits)} dígito(s) detectado(s) ·
    Confianza media: {resultado.mean_confidence * 100:.1f} %
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if resultado.fallback:
            st.warning(
                "No se detectaron dígitos separados. Se ha procesado la imagen "
                "completa como un único dígito. Para números de varios dígitos, "
                "deja espacio entre ellos."
            )

        if resultado.mean_confidence < 0.50:
            st.warning(
                "El modelo no está seguro del resultado. Revisa la nitidez, el "
                "contraste y que el número esté centrado en la imagen."
            )

        tabla = [
            {
                "Posición": item.index + 1,
                "Dígito": item.digit,
                "Confianza (%)": round(item.confidence * 100, 2),
            }
            for item in resultado.predictions
        ]
        st.dataframe(tabla, width="stretch", hide_index=True)

        st.markdown("#### Imagen original e imagen procesada")
        col_original, col_procesada = st.columns(2)
        with col_original:
            st.image(
                bgr_to_rgb(resultado.original_image),
                caption="Imagen original",
                width="stretch",
            )
        with col_procesada:
            st.image(
                resultado.binary_image,
                caption="Imagen procesada (blanco = trazo detectado)",
                width="stretch",
                clamp=True,
            )

        st.image(
            bgr_to_rgb(resultado.annotated_image),
            caption="Detección de cada dígito",
            width="stretch",
        )

        with st.expander("Ver cada dígito preparado para el modelo (28×28)"):
            columnas = st.columns(min(len(resultado.predictions), 5))
            for indice, item in enumerate(resultado.predictions):
                with columnas[indice % len(columnas)]:
                    st.image(
                        to_uint8(item.prepared),
                        caption=f"{item.digit} · {item.confidence * 100:.0f} %",
                        width="stretch",
                        clamp=True,
                    )

st.divider()
st.caption(
    "Proyecto académico de Machine Learning y visión por computadora. "
    "Las predicciones proceden de modelos entrenados con el dataset público "
    "MNIST; las imágenes que subes se procesan localmente."
)
