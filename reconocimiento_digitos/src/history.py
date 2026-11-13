"""Historial local de pruebas de reconocimiento.

Cada reconocimiento se guarda como un registro en un archivo JSON dentro de
``results/``. No se almacenan las imágenes: solo el nombre del archivo, el
modelo, el número reconocido y las confianzas.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import config

MAX_ENTRIES = 100

Record = dict[str, Any]


def _target(path: Path | None) -> Path:
    return Path(path) if path is not None else config.HISTORY_FILE


def load_history(path: Path | None = None) -> list[Record]:
    """Carga el historial.

    Devuelve una lista vacía si el archivo no existe, está dañado o no contiene
    una lista de registros.
    """
    target = _target(path)
    if not target.exists():
        return []
    try:
        with target.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def append_history(
    record: Record,
    path: Path | None = None,
    max_entries: int = MAX_ENTRIES,
) -> list[Record]:
    """Añade un registro al inicio del historial y guarda el archivo.

    La escritura se hace en un archivo temporal y luego se reemplaza el
    original, para evitar dejar un JSON incompleto si ocurre un error.
    """
    target = _target(path)
    history = load_history(target)
    entry = dict(record)
    entry.setdefault("fecha", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    history.insert(0, entry)
    history = history[:max_entries]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return history
