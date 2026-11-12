"""
Segmentación de dígitos manuscritos con OpenCV.

El objetivo es que el usuario pueda subir una foto con uno o varios números y
el sistema recorte cada dígito por separado, en orden de lectura (izquierda a
derecha), para alimentar al modelo de clasificación.

Limitaciones documentadas:
- Se asumen dígitos en una sola línea. Los números en varias líneas se ordenan
  globalmente por coordenada X, no por filas.
- Si los dígitos se tocan, se intenta una separación por perfil de proyección;
  si no es posible, se tratan como un único componente.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from src.preprocessing import (
    binarize_digit,
    enhance_contrast,
    ensure_grayscale,
    prepare_digit_for_model,
)


@dataclass
class SegmentedDigit:
    """Resultado de segmentar un dígito individual."""

    index: int
    bbox: tuple[int, int, int, int]  # x, y, ancho, alto en la imagen original
    binary_crop: np.ndarray          # máscara blanca sobre negro (uint8)
    prepared: np.ndarray             # imagen float32 28x28 lista para el modelo
    area: float


def _binarize_for_segmentation(gray: np.ndarray) -> np.ndarray:
    """Umbraliza y deja el trazo en blanco sobre fondo negro."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = binarize_digit(blurred)
    # Cierre suave para unir trazos que se cortaron por ruido, seguido de una
    # apertura para eliminar motas pequeñas.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def _split_wide_box(
    binary: np.ndarray,
    box: tuple[int, int, int, int],
    area: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Intenta separar dos dígitos que se tocaron y formaron un bloque ancho.

    Se analiza el perfil vertical de píxeles del componente. Si existe un valle
    claro cerca de la mitad, se divide en dos cajas. Es una heurística: no
    pretende resolver todos los casos de caligrafía unida.
    """
    x, y, w, h = box
    if w < 1.25 * h or w < 20:
        return [(box, area)]

    roi = binary[y : y + h, x : x + w]
    profile = (roi > 0).sum(axis=0).astype(np.float32)
    if profile.size < 10:
        return [(box, area)]

    # Suavizado del perfil para evitar cortar por un píxel aislado.
    window = max(3, int(profile.size * 0.05) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    smooth = np.convolve(profile, kernel, mode="same")

    left = int(profile.size * 0.30)
    right = int(profile.size * 0.70)
    if right <= left:
        return [(box, area)]

    center_region = smooth[left:right]
    split_offset = int(np.argmin(center_region)) + left
    valley = float(center_region.min())
    peak = float(smooth.max())
    if peak <= 0 or valley > 0.35 * peak:
        return [(box, area)]

    # Evitar divisiones que produzcan mitades demasiado estrechas.
    if split_offset < 0.2 * w or split_offset > 0.8 * w:
        return [(box, area)]

    left_w = split_offset
    right_w = w - split_offset
    left_area = float((roi[:, :split_offset] > 0).sum())
    right_area = float((roi[:, split_offset:] > 0).sum())
    return [
        ((x, y, left_w, h), left_area),
        ((x + split_offset, y, right_w, h), right_area),
    ]


def segment_digits(
    image: np.ndarray,
    min_area: int = config.DEFAULT_MIN_AREA,
    max_digits: int = config.DEFAULT_MAX_DIGITS,
    apply_clahe: bool = True,
) -> list[SegmentedDigit]:
    """Segmenta los dígitos de una imagen y devuelve sus recortes preparados.

    Parámetros
    ----------
    image:
        Imagen BGR, BGRA o en escala de grises.
    min_area:
        Área mínima de un contorno para considerarlo un dígito.
    max_digits:
        Número máximo de dígitos a devolver (protege la interfaz).
    apply_clahe:
        Mejora el contraste local antes de umbralizar (recomendado en fotos).
    """
    gray = ensure_grayscale(image)
    if apply_clahe:
        gray = enhance_contrast(gray)

    binary = _binarize_for_segmentation(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        if area < min_area or h < 8 or w < 2:
            continue
        candidates.append(((x, y, w, h), area))

    if not candidates:
        return []

    # Filtrar ruido comparando alturas y áreas con el componente principal.
    max_height = max(h for (_, _, _, h), _ in candidates)
    max_area = max(area for _, area in candidates)
    filtered = [
        (box, area)
        for box, area in candidates
        if box[3] >= 0.20 * max_height and area >= 0.01 * max_area
    ]

    # Intentar separar bloques anchos (dígitos pegados).
    expanded: list[tuple[tuple[int, int, int, int], float]] = []
    for box, area in filtered:
        expanded.extend(_split_wide_box(binary, box, area))

    # Orden de lectura: de izquierda a derecha.
    expanded.sort(key=lambda item: item[0][0])
    expanded = expanded[:max_digits]

    segments: list[SegmentedDigit] = []
    for index, ((x, y, w, h), area) in enumerate(expanded):
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(binary.shape[1], x + w + pad)
        y1 = min(binary.shape[0], y + h + pad)
        crop = binary[y0:y1, x0:x1]
        prepared = prepare_digit_for_model(
            crop,
            binarize=False,
            denoise=False,
        )
        segments.append(
            SegmentedDigit(
                index=index,
                bbox=(x, y, w, h),
                binary_crop=crop,
                prepared=prepared,
                area=area,
            )
        )
    return segments


def annotate_segments(image: np.ndarray, segments: list[SegmentedDigit]) -> np.ndarray:
    """Dibuja los recuadros y el número de orden sobre la imagen original."""
    gray = ensure_grayscale(image)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for segment in segments:
        x, y, w, h = segment.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (46, 204, 113), 2)
        label = str(segment.index + 1)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = max(0, x)
        ty = max(label_size[1] + 4, y - 6)
        cv2.rectangle(
            canvas,
            (tx, ty - label_size[1] - 4),
            (tx + label_size[0] + 6, ty + 2),
            (46, 204, 113),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (tx + 3, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def prepare_full_image(image: np.ndarray) -> np.ndarray:
    """Fallback: prepara la imagen completa como si fuera un único dígito."""
    return prepare_digit_for_model(image, binarize=True, denoise=True)


def segmentation_preview(image: np.ndarray) -> np.ndarray:
    """Devuelve la máscara binaria usada por el segmentador (para mostrar)."""
    gray = ensure_grayscale(image)
    gray = enhance_contrast(gray)
    return _binarize_for_segmentation(gray)
