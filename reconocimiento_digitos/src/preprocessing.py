"""
Preprocesamiento de imágenes para el reconocimiento de dígitos.

La cadena completa para una imagen subida por el usuario es:

1. Conversión a escala de grises.
2. Mejora de contraste (CLAHE) para fotos con iluminación desigual.
3. Eliminación de ruido (filtro de mediana / gaussiano).
4. Umbralización de Otsu robusta a polaridad (dedo oscuro sobre fondo claro
   o dedo claro sobre fondo oscuro).
5. Recorte al contenido, escalado conservando la relación de aspecto hasta
   20x20 y centrado por centro de masa en un lienzo de 28x28 (formato MNIST).
6. Normalización de píxeles a [0, 1].

Las imágenes de MNIST ya vienen centradas y en 28x28, por lo que para entrenar
solo se normalizan; no se les aplica umbralización ni redimensionado.
"""

from __future__ import annotations

import cv2
import numpy as np

import config


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR, BGRA o RGB a un array 2D en escala de grises.

    OpenCV lee las imágenes en orden BGR. Para imágenes de dígitos el orden de
    los canales no afecta de forma apreciable, pero se documenta la convención.
    """
    if image is None:
        raise ValueError("La imagen recibida es None.")
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("La imagen está vacía.")

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            # Componer el canal alfa sobre fondo blanco para no convertir en
            # negro las zonas transparentes de un PNG.
            bgr = arr[:, :, :3].astype(np.float32)
            alpha = (arr[:, :, 3:4].astype(np.float32) / 255.0)
            composed = bgr * alpha + 255.0 * (1.0 - alpha)
            gray = cv2.cvtColor(composed.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 3:
            gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif arr.shape[2] == 1:
            gray = arr[:, :, 0]
        else:
            raise ValueError(f"Número de canales no soportado: {arr.shape[2]}")
    else:
        raise ValueError(f"Dimensiones no soportadas: {arr.ndim}")

    return np.ascontiguousarray(gray.astype(np.uint8))


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Aplica CLAHE para atenuar sombras y mejorar el contraste local."""
    gray = ensure_grayscale(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise_image(image: np.ndarray, method: str = "median", kernel_size: int = 3) -> np.ndarray:
    """Elimina ruido de una imagen uint8.

    `median` es muy eficaz contra ruido de sal y pimienta, habitual en fotos.
    `gaussian` suaviza ruido gaussiano, pero difumina más los bordes.
    """
    gray = ensure_grayscale(image)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)
    if method == "median":
        return cv2.medianBlur(gray, kernel_size)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, 5, 50, 50)
    raise ValueError(f"Método de eliminación de ruido no soportado: {method}")


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Convierte píxeles a float32 en el rango [0, 1].

    Solo se divide entre 255 si los valores están claramente en escala 0-255.
    El umbral 1.5 evita dividir dos veces arrays normalizados que, tras una
    interpolación, pueden contener valores como 1.0000001.
    """
    source = np.asarray(image)
    arr = source.astype(np.float32)
    if arr.size and float(arr.max()) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def resize_digit(image: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Redimensiona una imagen a `size` x `size` (por defecto 28x28)."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        arr = ensure_grayscale(arr)
    interpolation = cv2.INTER_AREA if max(arr.shape) > size else cv2.INTER_LINEAR
    return cv2.resize(arr, (size, size), interpolation=interpolation)


def binarize_digit(image: np.ndarray, blur: bool = True) -> np.ndarray:
    """Umbraliza con Otsu y devuelve el dígito en blanco (255) sobre negro (0).

    La polaridad se elige automáticamente: de las dos posibles salidas de Otsu
    se conserva aquella en la que el "objeto" ocupa menos superficie, que es lo
    habitual para un dígito. Así funciona tanto con tinta oscura sobre papel
    claro como con tiza clara sobre pizarra oscura.
    """
    gray = ensure_grayscale(image)
    if blur:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    foreground_ratio = float(np.count_nonzero(binary)) / binary.size
    if foreground_ratio > 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def remove_small_objects(binary: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Elimina componentes conectados de área menor que `min_area`."""
    if binary.dtype != np.uint8:
        binary = (binary > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def crop_to_content(image: np.ndarray, margin: int = 0) -> np.ndarray:
    """Recorta al rectángulo que contiene los píxeles no nulos."""
    arr = np.asarray(image)
    coords = cv2.findNonZero((arr > 0).astype(np.uint8))
    if coords is None:
        return arr
    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(arr.shape[1], x + w + margin)
    y1 = min(arr.shape[0], y + h + margin)
    return arr[y0:y1, x0:x1]


def center_by_mass(image: np.ndarray, output_size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Desplaza la imagen para que su centro de masa quede en el centro.

    MNIST está normalizado así; replicarlo mejora mucho la precisión con
    imágenes nuevas, sobre todo si el usuario escribe en una esquina.
    """
    arr = normalize_pixels(image)
    if arr.shape != (output_size, output_size):
        arr = resize_digit(arr, output_size).astype(np.float32)

    total = float(arr.sum())
    if total <= 1e-8:
        return arr.astype(np.float32)

    ys, xs = np.indices(arr.shape, dtype=np.float32)
    cy = float((ys * arr).sum() / total)
    cx = float((xs * arr).sum() / total)
    shift_x = (output_size - 1) / 2.0 - cx
    shift_y = (output_size - 1) / 2.0 - cy

    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(
        arr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(shifted, 0.0, 1.0).astype(np.float32)


def resize_and_center(
    image: np.ndarray,
    box_size: int = config.MNIST_BOX_SIZE,
    output_size: int = config.IMAGE_SIZE,
) -> np.ndarray:
    """Escala el dígito a `box_size` conservando aspecto y lo centra en 28x28.

    Es la normalización estándar de MNIST: el dígito ocupa como máximo 20x20
    píxeles dentro de un lienzo de 28x28, con 4 píxeles de margen.
    """
    arr = normalize_pixels(image)

    # No se redimensiona aquí a 28x28: hacerlo antes de recortar deformaría
    # los dígitos estrechos (por ejemplo, un "1") y los convertiría en manchas.
    # Primero se recorta al contenido y luego se escala conservando el aspecto.
    content = crop_to_content(arr, margin=0)
    if content.size == 0 or np.count_nonzero(content) == 0:
        return np.zeros((output_size, output_size), dtype=np.float32)

    h, w = content.shape
    scale = float(box_size) / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(content, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((output_size, output_size), dtype=np.float32)
    x0 = (output_size - new_w) // 2
    y0 = (output_size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return center_by_mass(canvas, output_size=output_size)


def prepare_digit_for_model(
    image: np.ndarray,
    binarize: bool = True,
    denoise: bool = True,
    output_size: int = config.IMAGE_SIZE,
) -> np.ndarray:
    """Convierte una imagen de un dígito al formato de entrada de los modelos.

    Devuelve un array float32 de forma (28, 28) con el dígito blanco sobre
    fondo negro y valores en [0, 1].

    Parámetros
    ----------
    binarize:
        True para fotos o capturas. False si la imagen ya es una máscara
        binaria blanca sobre negro (por ejemplo, un recorte de segmentación).
    denoise:
        Aplica un filtro de mediana antes de umbralizar.
    """
    gray = ensure_grayscale(image)

    if binarize:
        if denoise:
            gray = denoise_image(gray, method="median", kernel_size=3)
        try:
            gray = enhance_contrast(gray)
        except cv2.error:  # imágenes demasiado pequeñas para CLAHE
            pass
        digit = binarize_digit(gray)
        digit_float = normalize_pixels(digit)
    else:
        digit_float = normalize_pixels(gray)

    prepared = resize_and_center(digit_float, output_size=output_size)
    return prepared.astype(np.float32)


def prepare_dataset_for_ml(X: np.ndarray) -> np.ndarray:
    """Aplana y normaliza un lote de imágenes para modelos clásicos.

    (N, 28, 28) uint8 -> (N, 784) float32 en [0, 1].
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4:
        X = X.squeeze(-1)
    if X.ndim == 3:
        X = X.reshape(len(X), -1)
    if X.size and X.max() > 1.0:
        X = X / 255.0
    return np.clip(X, 0.0, 1.0).astype(np.float32)


def prepare_dataset_for_cnn(X: np.ndarray) -> np.ndarray:
    """Prepara un lote para la CNN: (N, 28, 28, 1) float32 en [0, 1]."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 4 and X.shape[-1] == 1:
        out = X
    elif X.ndim == 3:
        out = X[..., np.newaxis]
    elif X.ndim == 2:
        out = X.reshape(len(X), config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
    else:
        raise ValueError(f"Forma no soportada para CNN: {X.shape}")
    if out.size and out.max() > 1.0:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen normalizada [0,1] a uint8 [0,255] para mostrarla."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_white_background(image: np.ndarray) -> np.ndarray:
    """Devuelve una versión del dígito con fondo blanco para visualización."""
    gray = ensure_grayscale(image)
    return 255 - gray
