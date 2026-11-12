"""
Generación de imágenes de demostración y prueba del pipeline completo.

IMPORTANTE: las imágenes se construyen a partir de dígitos del conjunto de
test de MNIST, aplicando rotaciones, ruido y desenfoque para simular una foto.
No son fotografías reales ni se presentan como tales. El objetivo es probar de
forma honesta la cadena segmentación -> preprocesamiento -> predicción con
imágenes que el modelo no vio durante el entrenamiento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

import config
from src.dataset import load_dataset
from src.prediction import predict_image
from src.training import load_trained_model, model_exists
from src.utils import save_json

MANIFEST_FILE = config.SAMPLES_DIR / "demo_manifest.json"

# Secuencias fijas para que la demostración sea interpretable. Cada dígito se
# toma de una imagen distinta del test de MNIST.
SINGLE_DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MULTI_SEQUENCES = {
    "numero_2026.png": [2, 0, 2, 6],
    "numero_31415.png": [3, 1, 4, 1, 5],
    "numero_789.png": [7, 8, 9],
}


def _sample_for_digit(
    X: np.ndarray,
    y: np.ndarray,
    digit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Selecciona una imagen aleatoria del test para un dígito concreto."""
    indices = np.flatnonzero(y == digit)
    if indices.size == 0:
        raise ValueError(f"No hay imágenes del dígito {digit} en el conjunto de test.")
    return X[int(rng.choice(indices))]


def _render_digit(
    digit_image: np.ndarray,
    target_height: int,
    rng: np.random.Generator,
) -> Image.Image:
    """Convierte una imagen MNIST en un trazo oscuro sobre fondo blanco."""
    inverted = (255 - digit_image).astype(np.uint8)
    pil = Image.fromarray(inverted, mode="L")
    scale = target_height / pil.height
    pil = pil.resize((max(1, int(round(pil.width * scale))), target_height), Image.LANCZOS)
    angle = float(rng.uniform(-7.0, 7.0))
    return pil.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def _apply_photo_effect(pil: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Añade ruido y un desenfoque leve para simular una captura con cámara."""
    arr = np.asarray(pil, dtype=np.float32)
    noise = rng.normal(0.0, 4.5, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr, mode="L")
    return result.filter(ImageFilter.GaussianBlur(radius=0.45))


def _compose(
    images: list[Image.Image],
    rng: np.random.Generator,
    gap: int,
    margin: int,
    canvas_height: int,
) -> Image.Image:
    """Pega varios dígitos sobre un lienzo blanco con pequeñas variaciones."""
    total_width = margin * 2 + sum(img.width for img in images) + gap * (len(images) - 1)
    canvas = Image.new("L", (total_width, canvas_height), color=255)

    x = margin
    for image in images:
        jitter_y = int(rng.integers(-6, 7))
        y = max(0, (canvas_height - image.height) // 2 + jitter_y)
        canvas.paste(image, (x, y))
        x += image.width + gap

    return _apply_photo_effect(canvas, rng)


def generate_demo_images(force: bool = True) -> list[dict[str, Any]]:
    """Crea las imágenes de demostración y su manifiesto.

    Returns
    -------
    list[dict]
        Elementos con `file`, `expected` y `kind`.
    """
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    if MANIFEST_FILE.exists() and not force:
        from src.utils import load_json

        return load_json(MANIFEST_FILE)

    data = load_dataset()
    X_test = data["X_test"]
    y_test = data["y_test"]
    rng = np.random.default_rng(config.RANDOM_STATE)

    # Dígitos individuales 0-9.
    for digit in SINGLE_DIGITS:
        digit_img = _sample_for_digit(X_test, y_test, digit, rng)
        rendered = _render_digit(digit_img, target_height=145, rng=rng)
        canvas = Image.new("L", (230, 230), color=255)
        x = (canvas.width - rendered.width) // 2 + int(rng.integers(-10, 11))
        y = (canvas.height - rendered.height) // 2 + int(rng.integers(-8, 9))
        canvas.paste(rendered, (max(0, x), max(0, y)))
        canvas = _apply_photo_effect(canvas, rng)

        filename = f"digito_{digit}.png"
        canvas.save(config.SAMPLES_DIR / filename)
        manifest.append(
            {
                "file": filename,
                "expected": str(digit),
                "kind": "single",
                "description": f"Dígito {digit} (MNIST test + rotación/ruido)",
            }
        )

    # Números de varios dígitos.
    for filename, sequence in MULTI_SEQUENCES.items():
        rendered = [
            _render_digit(_sample_for_digit(X_test, y_test, digit, rng), 150, rng)
            for digit in sequence
        ]
        canvas = _compose(
            rendered,
            rng=rng,
            gap=26,
            margin=38,
            canvas_height=235,
        )
        canvas.save(config.SAMPLES_DIR / filename)
        expected = "".join(str(d) for d in sequence)
        manifest.append(
            {
                "file": filename,
                "expected": expected,
                "kind": "multi",
                "description": f"Número {expected} (dígitos MNIST test compuestos)",
            }
        )

    save_json(MANIFEST_FILE, manifest)
    print(f"Imágenes de demostración generadas en {config.SAMPLES_DIR}")
    return manifest


def _choose_best_model() -> str | None:
    """Elige el mejor modelo disponible: CNN si existe; si no, el de mayor accuracy."""
    if model_exists("cnn"):
        return "cnn"
    if config.METRICS_FILE.exists():
        import pandas as pd

        df = pd.read_csv(config.METRICS_FILE)
        for name in df.sort_values("accuracy", ascending=False)["modelo"]:
            if name in config.MODEL_FILES and model_exists(str(name)):
                return str(name)
    for name in ("mlp", "svm", "knn", "logreg"):
        if model_exists(name):
            return name
    return None


def generate_demo_report(model_name: str | None = None) -> dict[str, Any]:
    """Ejecuta el pipeline completo sobre las muestras y guarda los resultados."""
    manifest = generate_demo_images(force=True)

    model_name = model_name or _choose_best_model()
    if model_name is None or not model_exists(model_name):
        print(
            "No hay modelos entrenados: se generaron las imágenes, pero no las "
            "predicciones. Ejecuta `python main.py train --model all`."
        )
        return {"model": None, "results": [], "sequence_accuracy": None}

    model = load_trained_model(model_name)
    results: list[dict[str, Any]] = []

    for item in manifest:
        path = config.SAMPLES_DIR / item["file"]
        prediction = predict_image(path, model, model_name)
        expected = str(item["expected"])
        correct = prediction.number == expected
        results.append(
            {
                "file": item["file"],
                "expected": expected,
                "predicted": prediction.number,
                "correct": correct,
                "confidence": round(prediction.mean_confidence, 6),
                "digits": prediction.digits,
                "confidences": [round(c, 6) for c in prediction.confidences],
                "fallback": prediction.fallback,
            }
        )
        annotated_path = config.PLOTS_DIR / f"demo_annotated_{Path(item['file']).stem}.png"
        cv2.imwrite(str(annotated_path), prediction.annotated_image)

    sequence_accuracy = float(np.mean([r["correct"] for r in results]))
    payload = {
        "model": model_name,
        "model_display": config.MODEL_DISPLAY_NAMES[model_name],
        "note": (
            "Imágenes sintéticas compuestas a partir del test de MNIST con "
            "rotación, ruido y desenfoque. No son fotografías reales."
        ),
        "sequence_accuracy": sequence_accuracy,
        "results": results,
    }
    save_json(config.DEMO_RESULTS_FILE, payload)

    _plot_demo(results, model_name)
    print(
        f"Demostración completada con {config.MODEL_DISPLAY_NAMES[model_name]}: "
        f"{sum(r['correct'] for r in results)}/{len(results)} secuencias correctas "
        f"({sequence_accuracy * 100:.1f} %)."
    )
    return payload


def _plot_demo(results: list[dict[str, Any]], model_name: str) -> None:
    """Guarda una figura comparando imagen original y predicción anotada."""
    selected = results[:4]
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.1 * len(selected)))
    axes = np.atleast_2d(axes)

    for row, result in enumerate(selected):
        path = config.SAMPLES_DIR / result["file"]
        original = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        annotated_path = config.PLOTS_DIR / f"demo_annotated_{Path(result['file']).stem}.png"
        annotated = cv2.cvtColor(cv2.imread(str(annotated_path)), cv2.COLOR_BGR2RGB)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Original · esperado: {result['expected']}", fontsize=10)
        axes[row, 0].axis("off")

        color = "#15803d" if result["correct"] else "#b91c1c"
        axes[row, 1].imshow(annotated)
        axes[row, 1].set_title(
            f"Predicción: {result['predicted']} "
            f"(conf. media {result['confidence'] * 100:.1f} %)",
            fontsize=10,
            color=color,
        )
        axes[row, 1].axis("off")

    fig.suptitle(
        f"Demostración del pipeline completo · {config.MODEL_DISPLAY_NAMES[model_name]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "demo_predictions.png", bbox_inches="tight")
    plt.close(fig)
