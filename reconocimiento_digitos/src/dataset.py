"""
Carga, división y almacenamiento del dataset MNIST.

Decisiones de diseño:
- Se usa la división oficial de Keras: 60 000 imágenes de entrenamiento y
  10 000 de prueba. El conjunto de prueba NUNCA se usa para entrenar ni para
  ajustar hiperparámetros.
- Del bloque de 60 000 se extrae un 10 % estratificado (6 000 imágenes) como
  validación. Así quedan 54 000 / 6 000 / 10 000 sin solapamiento.
- El resultado se cachea en `data/mnist_split.npz` para no descargar ni volver
  a dividir en cada ejecución.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.utils import save_json


def _validate_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Comprueba formas, tipos y rangos antes de guardar el dataset."""
    expected_train = (60_000, 28, 28)
    expected_test = (10_000, 28, 28)
    if X_train.shape != expected_train or X_test.shape != expected_test:
        raise ValueError(
            "Formas inesperadas de MNIST: "
            f"train={X_train.shape}, test={X_test.shape}. "
            "Se esperaban 60 000 y 10 000 imágenes de 28x28."
        )
    if y_train.shape[0] != X_train.shape[0] or y_test.shape[0] != X_test.shape[0]:
        raise ValueError("El número de etiquetas no coincide con el de imágenes.")
    if not (0 <= int(y_train.min()) and int(y_train.max()) <= 9):
        raise ValueError("Las etiquetas deben estar en el rango 0-9.")
    if X_train.dtype != np.uint8 or X_test.dtype != np.uint8:
        raise ValueError("Se esperaban imágenes uint8 (0-255).")


def download_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Descarga MNIST mediante Keras y devuelve los cuatro arrays originales.

    Keras guarda la caché en `~/.keras/datasets/mnist.npz`.
    """
    # Import local: así `prepare` puede mostrar un error amigable si TensorFlow
    # no está disponible, sin romper el resto de comandos del proyecto.
    from tensorflow import keras

    try:
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as exc:  # pragma: no cover - depende de la red
        raise RuntimeError(
            "No se pudo descargar MNIST. Comprueba tu conexión a Internet y "
            "vuelve a ejecutar `python main.py prepare`. Keras descarga el "
            "archivo desde https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        ) from exc

    _validate_arrays(X_train, y_train, X_test, y_test)
    return X_train, y_train, X_test, y_test


def _save_labels(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> None:
    """Guarda las etiquetas en CSV, una fila por imagen y su índice original."""
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": y_train,
        "validation": y_val,
        "test": y_test,
    }
    for split_name, labels in splits.items():
        df = pd.DataFrame(
            {
                "indice": np.arange(len(labels), dtype=np.int64),
                "etiqueta": labels.astype(np.int64),
            }
        )
        df.to_csv(config.LABELS_DIR / f"{split_name}_labels.csv", index=False)


def split_dataset(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, np.ndarray]:
    """Divide el bloque de entrenamiento en train/validación de forma estratificada.

    La estratificación conserva la proporción de cada dígito. `random_state`
    fijo hace la división reproducible.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.VALIDATION_SIZE,
        stratify=y_train_full,
        random_state=config.RANDOM_STATE,
    )
    return {
        "X_train": np.ascontiguousarray(X_train),
        "y_train": np.ascontiguousarray(y_train),
        "X_val": np.ascontiguousarray(X_val),
        "y_val": np.ascontiguousarray(y_val),
        "X_test": np.ascontiguousarray(X_test),
        "y_test": np.ascontiguousarray(y_test),
    }


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    """Cuenta cuántas imágenes hay de cada dígito."""
    counts = np.bincount(y, minlength=config.NUM_CLASSES)
    return {str(i): int(counts[i]) for i in range(config.NUM_CLASSES)}


def build_dataset_info(splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Construye los metadatos que se muestran en consola, README y Streamlit."""
    return {
        "nombre": "MNIST (Modified National Institute of Standards and Technology)",
        "descripcion": (
            "Dígitos manuscritos en escala de grises, centrados y normalizados "
            "por el creador del dataset."
        ),
        "fuente": "TensorFlow/Keras: keras.datasets.mnist.load_data()",
        "fecha_preparacion": datetime.now().isoformat(timespec="seconds"),
        "dimensiones_imagen": [28, 28],
        "canales": 1,
        "valores_pixel": [0, 255],
        "clases": config.CLASS_NAMES,
        "semilla": config.RANDOM_STATE,
        "estrategia_division": (
            "División oficial train/test de MNIST (60 000/10 000) y separación "
            "estratificada del 10 % del train como validación (6 000). "
            "El test oficial permanece intacto."
        ),
        "conteos": {
            "train": int(len(splits["y_train"])),
            "validation": int(len(splits["y_val"])),
            "test": int(len(splits["y_test"])),
            "total": int(
                len(splits["y_train"]) + len(splits["y_val"]) + len(splits["y_test"])
            ),
        },
        "distribucion_train": _class_distribution(splits["y_train"]),
        "distribucion_validation": _class_distribution(splits["y_val"]),
        "distribucion_test": _class_distribution(splits["y_test"]),
        "formas": {
            "X_train": list(splits["X_train"].shape),
            "X_val": list(splits["X_val"].shape),
            "X_test": list(splits["X_test"].shape),
        },
        "dtype": "uint8",
        "nota": (
            "Las métricas finales se calculan únicamente sobre X_test. "
            "La validación se usa para early stopping y ajuste de épocas."
        ),
    }


def prepare_dataset(force: bool = False) -> dict[str, np.ndarray]:
    """Descarga (si hace falta), divide, guarda y devuelve el dataset.

    Parámetros
    ----------
    force:
        Si es True, ignora la caché y vuelve a generar la división.
    """
    config.ensure_directories()

    if config.DATA_FILE.exists() and not force:
        return load_dataset()

    print("Descargando MNIST desde Keras (puede tardar la primera vez)...")
    X_train_full, y_train_full, X_test, y_test = download_mnist()
    print("MNIST descargado. Separando entrenamiento y validación...")
    splits = split_dataset(X_train_full, y_train_full, X_test, y_test)

    np.savez_compressed(
        config.DATA_FILE,
        **splits,
    )
    _save_labels(splits["y_train"], splits["y_val"], splits["y_test"])

    info = build_dataset_info(splits)
    save_json(config.DATASET_INFO_FILE, info)
    print(f"Dataset guardado en: {config.DATA_FILE}")
    return splits


def load_dataset() -> dict[str, np.ndarray]:
    """Carga la división cacheada; si no existe, la prepara."""
    if not config.DATA_FILE.exists():
        return prepare_dataset()

    with np.load(config.DATA_FILE, allow_pickle=False) as data:
        splits = {key: data[key] for key in data.files}

    required = {"X_train", "y_train", "X_val", "y_val", "X_test", "y_test"}
    missing = required.difference(splits)
    if missing:
        raise ValueError(f"El archivo de dataset está incompleto: faltan {missing}")
    return splits


def load_dataset_info() -> dict[str, Any]:
    """Devuelve los metadatos del dataset (o los reconstruye si faltan)."""
    if config.DATASET_INFO_FILE.exists():
        import json

        with config.DATASET_INFO_FILE.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return build_dataset_info(load_dataset())


def describe_dataset(splits: dict[str, np.ndarray]) -> pd.DataFrame:
    """Crea una tabla resumen de los tres conjuntos."""
    rows = []
    for name, key in (
        ("Entrenamiento", "X_train"),
        ("Validación", "X_val"),
        ("Prueba", "X_test"),
    ):
        X = splits[key]
        y = splits[key.replace("X_", "y_")]
        rows.append(
            {
                "Conjunto": name,
                "Imágenes": len(X),
                "Forma": f"{X.shape[1]}x{X.shape[2]}",
                "Canales": 1,
                "Etiquetas": len(y),
                "Clases": len(np.unique(y)),
            }
        )
    return pd.DataFrame(rows)


def dataset_paths() -> list[Path]:
    """Lista los archivos generados por la preparación del dataset."""
    return [
        config.DATA_FILE,
        config.DATASET_INFO_FILE,
        config.LABELS_DIR / "train_labels.csv",
        config.LABELS_DIR / "validation_labels.csv",
        config.LABELS_DIR / "test_labels.csv",
    ]
