"""Utilidades compartidas: temporización, JSON, submuestreo estratificado."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import train_test_split


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager que mide tiempo transcurrido en segundos.

    Uso:
        with timer() as t:
            ...
        print(t["seconds"])
    """
    start = time.perf_counter()
    info: dict[str, float] = {}
    try:
        yield info
    finally:
        info["seconds"] = round(time.perf_counter() - start, 3)


def save_json(path: Path, data: Any) -> None:
    """Guarda datos serializables en JSON con codificación UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    """Carga un JSON; lanza FileNotFoundError si no existe."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    """Convierte tipos de NumPy a tipos nativos para poder serializar."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Objeto no serializable: {type(obj)!r}")


def stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    size: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve un subconjunto estratificado de tamaño `size` (o todo si es menor).

    Se usa para SVM y validación cruzada, donde entrenar con las 54 000
    imágenes completas resultaría innecesariamente costoso en CPU.
    """
    if size >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=size,
        stratify=y,
        random_state=random_state,
    )
    return X_sub, y_sub


def format_seconds(seconds: float) -> str:
    """Formatea una duración en un texto legible."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} min {rest:.0f} s"


def human_bytes(num_bytes: int) -> str:
    """Convierte bytes a una cadena legible."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"
