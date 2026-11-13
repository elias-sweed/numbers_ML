"""
Inferencia sobre imágenes nuevas: carga, segmentación, predicción y confianza.

El flujo es independiente del modelo elegido:
1. Cargar la imagen (ruta, bytes o array) y corregir la orientación EXIF.
2. Segmentar los dígitos con OpenCV.
3. Preparar cada recorte al formato MNIST (28x28, centrado por masa).
4. Obtener probabilidades del modelo.
5. Devolver dígitos, número completo, confianzas e imágenes de apoyo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

import config
from src.preprocessing import (
    binarize_digit,
    ensure_grayscale,
    prepare_dataset_for_cnn,
    prepare_dataset_for_ml,
    prepare_digit_for_model,
)
from src.segmentation import (
    SegmentedDigit,
    annotate_segments,
    segment_digits,
    segmentation_preview,
)

ImageSource = Union[str, Path, bytes, bytearray, np.ndarray, Image.Image]


# ---------------------------------------------------------------------------
# Carga de imágenes
# ---------------------------------------------------------------------------
def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL a BGR, componiendo transparencia sobre blanco."""
    pil = ImageOps.exif_transpose(pil)
    if pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info):
        rgba = pil.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        pil = pil.convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def load_image(source: ImageSource) -> np.ndarray:
    """Carga una imagen y la devuelve en BGR uint8.

    Acepta rutas, bytes (por ejemplo de Streamlit) o arrays. Usa PIL para
    respetar la orientación EXIF de las fotos de móvil y para componer fondos
    transparentes sobre blanco.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            # Componer el alfa sobre blanco (BGRA -> BGR).
            bgr = arr[:, :, :3].astype(np.float32)
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            arr = np.clip(bgr * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr.astype(np.uint8))

    if isinstance(source, Image.Image):
        return _pil_to_bgr(source)

    if isinstance(source, (bytes, bytearray)):
        buffer = BytesIO(bytes(source))
        pil = Image.open(buffer)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró la imagen: {path}")
        pil = Image.open(path)

    return _pil_to_bgr(pil)


# ---------------------------------------------------------------------------
# Probabilidades
# ---------------------------------------------------------------------------
def softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable por filas."""
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_probabilities(
    model: Any,
    model_name: str,
    prepared_images: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Predice probabilidades para una lista de imágenes de 28x28.

    Para el SVM (sin `predict_proba`) se aplica softmax a los márgenes de
    decisión. Es una confianza relativa, no una probabilidad calibrada.
    """
    if isinstance(prepared_images, np.ndarray) and prepared_images.ndim == 2:
        prepared_images = [prepared_images]
    batch = np.stack([np.asarray(img, dtype=np.float32) for img in prepared_images])

    if model_name == "cnn":
        X = prepare_dataset_for_cnn(batch)
        probs = model.predict(X, verbose=0)
    elif model_name == "mlp":
        # La MLP es un modelo Keras, pero espera la imagen aplanada (784).
        X = prepare_dataset_for_ml(batch)
        probs = model.predict(X, verbose=0)
    else:
        X = prepare_dataset_for_ml(batch)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        else:
            probs = softmax(model.decision_function(X))

    return np.asarray(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@dataclass
class DigitPrediction:
    """Predicción de un dígito individual."""

    index: int
    digit: int
    confidence: float
    probabilities: list[float]
    bbox: tuple[int, int, int, int] | None = None
    prepared: np.ndarray | None = None


@dataclass
class ImagePrediction:
    """Resultado completo de procesar una imagen con uno o varios dígitos."""

    number: str
    digits: list[int]
    confidences: list[float]
    mean_confidence: float
    predictions: list[DigitPrediction]
    original_image: np.ndarray
    annotated_image: np.ndarray
    binary_image: np.ndarray
    fallback: bool
    segments: list[SegmentedDigit] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.digits) == 0


def _annotate_predictions(
    image: np.ndarray,
    segments: list[SegmentedDigit],
    digits: list[int],
    confidences: list[float],
) -> np.ndarray:
    """Dibuja cada recorte con el dígito predicho y su confianza."""
    canvas = annotate_segments(image, segments)
    for segment, digit, confidence in zip(segments, digits, confidences):
        x, y, w, h = segment.bbox
        label = f"{digit} ({confidence * 100:.0f}%)"
        scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        top = max(0, y - text_h - 10)
        cv2.rectangle(
            canvas,
            (x, top),
            (x + text_w + 8, top + text_h + 8),
            (37, 99, 235),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x + 4, top + text_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def predict_image(
    source: ImageSource,
    model: Any,
    model_name: str,
    min_area: int = config.DEFAULT_MIN_AREA,
    max_digits: int = config.DEFAULT_MAX_DIGITS,
) -> ImagePrediction:
    """Segmenta y predice todos los dígitos de una imagen.

    Si no se detecta ningún componente, se procesa la imagen completa como un
    único dígito (comportamiento de respaldo, `fallback=True`).
    """
    image = load_image(source)
    segments = segment_digits(image, min_area=min_area, max_digits=max_digits)
    binary = segmentation_preview(image)

    if segments:
        prepared = [segment.prepared for segment in segments]
        fallback = False
    else:
        prepared = [prepare_digit_for_model(image, binarize=True, denoise=True)]
        fallback = True

    probabilities = predict_probabilities(model, model_name, prepared)
    digits = [int(value) for value in np.argmax(probabilities, axis=1)]
    confidences = [float(value) for value in probabilities.max(axis=1)]

    predictions = []
    for i, (digit, confidence) in enumerate(zip(digits, confidences)):
        segment = segments[i] if not fallback else None
        predictions.append(
            DigitPrediction(
                index=i,
                digit=digit,
                confidence=confidence,
                probabilities=[float(v) for v in probabilities[i]],
                bbox=segment.bbox if segment else None,
                prepared=prepared[i],
            )
        )

    if segments:
        annotated = _annotate_predictions(image, segments, digits, confidences)
    else:
        annotated = cv2.cvtColor(ensure_grayscale(image), cv2.COLOR_GRAY2BGR)
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (2, 2), (w - 3, h - 3), (37, 99, 235), 2)

    number = "".join(str(d) for d in digits)
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0

    return ImagePrediction(
        number=number,
        digits=digits,
        confidences=confidences,
        mean_confidence=mean_confidence,
        predictions=predictions,
        original_image=image,
        annotated_image=annotated,
        binary_image=binary,
        fallback=fallback,
        segments=segments,
    )


def predict_single_digit(
    source: ImageSource,
    model: Any,
    model_name: str,
) -> DigitPrediction:
    """Predice un único dígito sin segmentar (útil para pruebas y API)."""
    image = load_image(source)
    prepared = prepare_digit_for_model(image, binarize=True, denoise=True)
    probabilities = predict_probabilities(model, model_name, prepared)[0]
    digit = int(np.argmax(probabilities))
    return DigitPrediction(
        index=0,
        digit=digit,
        confidence=float(probabilities[digit]),
        probabilities=[float(v) for v in probabilities],
        prepared=prepared,
    )


def available_models() -> dict[str, bool]:
    """Indica qué modelos están entrenados y disponibles en disco."""
    return {
        name: path.exists()
        for name, path in config.MODEL_FILES.items()
        if name != "kmeans"
    }
