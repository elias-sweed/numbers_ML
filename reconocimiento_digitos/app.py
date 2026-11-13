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

import os

# Keras puede ejecutar los modelos con el backend JAX. Así la app funciona en
# Python 3.12, 3.13 y 3.14 sin instalar TensorFlow (que no tiene wheels para
# Python 3.14). El entrenamiento sigue usando TensorFlow.
os.environ.setdefault("KERAS_BACKEND", "jax")

from typing import Any

import cv2
import numpy as np
import streamlit as st

import config
from src.history import append_history, load_history
from src.prediction import ImagePrediction, available_models, predict_image
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


def imagen_sin_trazo(resultado: ImagePrediction) -> bool:
    """Indica si la imagen procesada no contiene suficientes píxeles de trazo."""
    return float(np.mean(resultado.binary_image > 0)) < 0.001


def explicar_resultado(resultado: ImagePrediction, modelo: str) -> str:
    """Redacta una explicación sencilla del resultado y de su confianza."""
    nombre_modelo = config.MODEL_DISPLAY_NAMES[modelo]
    total = len(resultado.predictions)
    confianza = resultado.mean_confidence * 100

    if resultado.fallback:
        deteccion = (
            "2. **No se encontraron varios trazos separados.** Por eso se tomó "
            "la imagen completa como si fuera **un solo dígito**. Para números "
            "de varias cifras, deja un espacio claro entre los dígitos."
        )
    elif total == 1:
        deteccion = (
            "2. **Se separó el dígito.** Se encontró **1 dígito** y se preparó "
            "para enviarlo al modelo."
        )
    else:
        deteccion = (
            f"2. **Se separaron los dígitos.** Se encontraron **{total} dígitos** "
            "y se ordenaron de izquierda a derecha, como se lee un número."
        )

    if confianza >= 90:
        nivel = "alta"
        valoracion = (
            "El modelo está muy seguro de su respuesta. Aun así, conviene "
            "revisar el resultado si la letra es muy irregular."
        )
    elif confianza >= 70:
        nivel = "media"
        valoracion = (
            "El modelo tiene una seguridad razonable, pero no total. Revisa la "
            "imagen si el número esperado era distinto."
        )
    else:
        nivel = "baja"
        valoracion = (
            "El modelo no está seguro. Revisa la nitidez, el contraste y que "
            "los dígitos estén bien separados antes de confiar en el resultado."
        )

    partes = [
        "#### ¿Cómo se obtuvo este resultado?",
        (
            "1. **Se preparó la imagen.** El sistema la convirtió a blanco y "
            "negro, mejoró el contraste y separó el trazo del fondo."
        ),
        deteccion,
        (
            "3. **Se clasificó cada recorte.** Cada dígito se redujo a una "
            "imagen pequeña de 28×28 píxeles y se envió al modelo "
            f"**{nombre_modelo}**, que eligió la clase más probable entre 0 y 9."
        ),
        (
            "4. **Se formó el número.** Al unir las respuestas en orden se "
            f"obtuvo **{resultado.number}**. El sistema reconoce la secuencia "
            "de dígitos; no interpreta su significado ni hace operaciones "
            "matemáticas."
        ),
        (
            f"**Confianza media: {confianza:.1f} % ({nivel}).** {valoracion} "
            "La confianza indica qué tan seguro está el modelo de su propia "
            "respuesta; no es una garantía de que el número sea correcto."
        ),
    ]
    if modelo == "svm":
        partes.append(
            "En el caso del SVM, este valor es una estimación relativa, no una "
            "probabilidad calibrada."
        )
    if total > 1:
        peor = min(resultado.predictions, key=lambda item: item.confidence)
        partes.append(
            "El dígito con menor confianza está en la posición "
            f"**{peor.index + 1}** (un **{peor.digit}**, "
            f"{peor.confidence * 100:.1f} %). Si el número no coincide, revisa "
            "primero ese dígito."
        )
    partes.append(
        "El sistema trabaja con **una línea de dígitos**. Si los trazos se tocan "
        "o la foto tiene sombras, la separación puede fallar."
    )
    return "\n\n".join(partes)


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

col_titulo, col_ayuda = st.columns([0.9, 0.1], vertical_alignment="center")
with col_titulo:
    st.subheader("1. Selecciona el modelo")
with col_ayuda:
    with st.popover("❓", help="¿Qué hace cada modelo?"):
        st.markdown(
            """
**Regresión Logística**  
Traza una frontera lineal entre los dígitos. Es rápida y fácil de interpretar.

**KNN (K-Nearest Neighbors)**  
Compara la imagen con ejemplos ya conocidos y toma la opinión de los más parecidos.

**SVM (Máquina de Vectores de Soporte)**  
Busca la mejor frontera de separación entre clases. Suele ser preciso, aunque más lento.

**MLP (Perceptrón Multicapa)**  
Red neuronal de capas densas que aprende combinaciones de píxeles.

**CNN (Red Neuronal Convolucional)**  
Red especializada en imágenes: detecta trazos y formas. Es la recomendada para dígitos manuscritos.
"""
        )
        st.caption(
            "Los cinco modelos están entrenados con MNIST y se cargan desde "
            "la carpeta models/."
        )

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
st.info(
    "**Qué imagen debes cargar**\n\n"
    "- Una foto o captura de **un número escrito a mano**, por ejemplo `405`.\n"
    "- **Una sola línea** de dígitos, con espacio entre ellos.\n"
    "- Trazo **oscuro sobre fondo claro**: papel blanco, tinta negra o azul.\n"
    "- Buena iluminación, sin sombras fuertes, arrugas ni reflejos.\n"
    "- Recorta la imagen para que el número ocupe la mayor parte.\n"
    "- Formatos admitidos: **PNG, JPG, JPEG, BMP y WEBP**.\n"
    "- No subas formularios completos, varias filas, letras ni dibujos."
)
archivo = st.file_uploader(
    "Imagen con uno o varios dígitos manuscritos",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
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
        sin_trazo_resultado = imagen_sin_trazo(prediccion)
        st.session_state["resultado"] = prediccion
        st.session_state["resultado_modelo"] = modelo
        st.session_state["resultado_archivo"] = archivo.name
        registro = {
            "archivo": archivo.name,
            "archivo_bytes": archivo.size,
            "modelo": config.MODEL_DISPLAY_NAMES[modelo],
            "modelo_id": modelo,
            "numero": prediccion.number,
            "confianza_media_pct": round(prediccion.mean_confidence * 100, 2),
            "digitos": len(prediccion.predictions),
            "fallback": bool(prediccion.fallback),
            "sin_trazo": sin_trazo_resultado,
            "detalle": [
                {
                    "posicion": item.index + 1,
                    "digito": item.digit,
                    "confianza_pct": round(item.confidence * 100, 2),
                }
                for item in prediccion.predictions
            ],
        }
        try:
            append_history(registro)
        except OSError as exc:
            st.warning(
                "El resultado se muestra, pero no se pudo guardar en el "
                f"historial. Detalle: {exc}"
            )
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

    sin_trazo = imagen_sin_trazo(resultado)

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

        with st.container(border=True):
            st.markdown(explicar_resultado(resultado, modelo))

        st.markdown("#### Confianza de cada dígito")
        st.caption(
            "Cuanto mayor sea el porcentaje, más segura es la respuesta para "
            "ese dígito."
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

historial = load_history()
with st.expander(f"Historial de pruebas ({len(historial)})"):
    if not historial:
        st.caption(
            "Todavía no hay pruebas guardadas. Cada reconocimiento se añadirá "
            "aquí automáticamente."
        )
    else:
        filas_historial = []
        for entrada in historial:
            sin_trazo_entrada = bool(entrada.get("sin_trazo"))
            confianza_entrada = entrada.get("confianza_media_pct")
            if sin_trazo_entrada:
                numero = "—"
                confianza_texto = "—"
                estado = "Sin trazo"
            else:
                numero = str(entrada.get("numero") or "—")
                confianza_texto = (
                    f"{float(confianza_entrada):.1f}"
                    if isinstance(confianza_entrada, (int, float))
                    else "—"
                )
                estado = "Procesado"
            filas_historial.append(
                {
                    "Fecha": entrada.get("fecha", ""),
                    "Imagen": entrada.get("archivo", ""),
                    "Modelo": entrada.get("modelo", ""),
                    "Número": numero,
                    "Confianza media (%)": confianza_texto,
                    "Dígitos": entrada.get("digitos", 0),
                    "Estado": estado,
                }
            )
        st.dataframe(filas_historial, width="stretch", hide_index=True)
        ruta_historial = config.HISTORY_FILE.relative_to(config.BASE_DIR).as_posix()
        st.caption(
            f"Se guardan hasta 100 pruebas en `{ruta_historial}`. "
            "La imagen no se almacena; solo su nombre y el resultado."
        )

st.caption(
    "Proyecto académico de Machine Learning y visión por computadora. "
    "Las predicciones proceden de modelos entrenados con el dataset público "
    "MNIST; las imágenes que subes se procesan localmente."
)
