"""
Prueba con fotografías reales de números manuscritos en papel.

Este módulo NO genera fotografías ni presenta imágenes sintéticas como reales.
Lee un manifiesto CSV con fotografías aportadas por el usuario, ejecuta el
pipeline completo y guarda los resultados de forma separada.

Estructura esperada:

    data/real_photos/
        manifest.csv
        foto_2026.jpg
        foto_31415.jpg
        ...

Columnas del manifiesto:
    archivo      Nombre o ruta de la fotografía.
    esperado     Número esperado, por ejemplo 2026. Puede quedar vacío.
    modelo       Modelo a usar; si está vacío se usa el de la CLI.
    min_area     Área mínima de segmentación; si está vacío se usa la CLI.
    tipo         Debe ser "foto_real". Otras filas se rechazan.
    descripcion  Nota libre sobre la foto.

Si no hay fotografías, el comando `python main.py real-test` explica cómo
añadirlas y no inventa ningún resultado.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import config
from src.prediction import predict_image
from src.training import load_trained_model, model_exists
from src.utils import save_json

REAL_PHOTOS_DIR = config.DATA_DIR / "real_photos"
DEFAULT_MANIFEST = REAL_PHOTOS_DIR / "manifest.csv"
ANNOTATED_DIR = config.RESULTS_DIR / "real_photos_anotadas"
REAL_RESULTS_JSON = config.RESULTS_DIR / "real_photos_results.json"
REAL_RESULTS_MD = config.RESULTS_DIR / "REAL_PHOTOS.md"

MANIFEST_COLUMNS = [
    "archivo",
    "esperado",
    "modelo",
    "min_area",
    "tipo",
    "descripcion",
]

README_TEXT = """# Fotografías reales

Esta carpeta es para **fotografías reales** de números escritos a mano en papel.
No debe contener imágenes sintéticas generadas por `src/demo.py` ni capturas de
MNIST.

## Cómo añadir una fotografía

1. Escribe un número en papel blanco con tinta oscura.
2. Haz una foto con el móvil, con buena luz y sin sombras fuertes.
3. Recorta la zona del número si la foto es muy grande.
4. Guarda la imagen en esta carpeta, por ejemplo `foto_2026.jpg`.
5. Añade una fila a `manifest.csv`:

```csv
archivo,esperado,modelo,min_area,tipo,descripcion
foto_2026.jpg,2026,cnn,40,foto_real,Número 2026 en papel blanco
```

Si no conoces el valor esperado, deja `esperado` vacío. Si no indicas modelo o
área mínima, se usan los valores de la CLI.

## Ejecutar la prueba

```powershell
cd C:\\Users\\elias\\Documents\\SARA\\reconocimiento_digitos
..\\.venv\\Scripts\\python.exe main.py real-test
```

También puedes probar una foto directamente en la interfaz web de Streamlit,
con el botón **Reconocer dígitos**.

## Reglas de honestidad

- Solo se aceptan filas con `tipo=foto_real`.
- Las imágenes de `data/samples/` son sintéticas y no cuentan como fotos reales.
- Si no hay fotos, el informe debe indicar que la prueba real está pendiente.
"""


def ensure_real_photo_workspace(force_template: bool = False) -> Path:
    """Crea la carpeta, las instrucciones y la plantilla de manifiesto."""
    REAL_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    readme = REAL_PHOTOS_DIR / "LEEME.md"
    if force_template or not readme.exists():
        readme.write_text(README_TEXT, encoding="utf-8")
    if force_template or not DEFAULT_MANIFEST.exists():
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(DEFAULT_MANIFEST, index=False)
    return DEFAULT_MANIFEST


def load_manifest(manifest_path: Path | None = None) -> pd.DataFrame:
    """Carga y valida el manifiesto de fotografías reales."""
    path = Path(manifest_path) if manifest_path else ensure_real_photo_workspace()
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto: {path}. "
            "Crea data/real_photos/manifest.csv con las columnas indicadas."
        )
    df = pd.read_csv(path)
    missing = [column for column in MANIFEST_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en {path}: {', '.join(missing)}. "
            f"Columnas requeridas: {', '.join(MANIFEST_COLUMNS)}"
        )
    return df


def _resolve_photo(row: pd.Series, manifest_path: Path) -> Path:
    """Resuelve una ruta relativa al manifiesto o una ruta absoluta."""
    raw = str(row["archivo"]).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _safe_stem(text: str) -> str:
    """Convierte un nombre de archivo en un identificador seguro."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
    return cleaned.strip("_") or "foto"


def _relative(path: Path) -> str:
    """Devuelve una ruta relativa al proyecto si es posible."""
    try:
        return str(path.relative_to(config.BASE_DIR))
    except ValueError:
        return str(path)


def _write_markdown(payload: dict[str, Any]) -> Path:
    """Escribe el informe de fotografías reales."""
    lines = [
        "# Prueba con fotografías reales",
        "",
        f"Generado: {payload['fecha']}",
        "",
        "> Solo se incluyen filas con `tipo=foto_real`. Las imágenes sintéticas",
        "> de `data/samples/` no cuentan como fotografías reales.",
        "",
    ]
    if payload["estado"] == "sin_fotografias":
        lines += [
            "## Estado: prueba pendiente",
            "",
            "No se encontraron fotografías reales en el manifiesto.",
            "",
            "Para completar esta prueba:",
            "",
            "1. Haz una foto real de un número escrito en papel.",
            "2. Guárdala en `data/real_photos/`.",
            "3. Añádela a `data/real_photos/manifest.csv` con `tipo=foto_real`.",
            "4. Ejecuta `python main.py real-test`.",
            "",
            "También puedes subir la foto directamente en la interfaz web de",
            "Streamlit.",
            "",
        ]
    else:
        lines += [
            "## Resumen",
            "",
            f"- Fotografías procesadas: {payload['n_fotos']}.",
            f"- Modelo: {payload['modelo_principal']}.",
            f"- Secuencias correctas: {payload['secuencias_correctas']}/{payload['n_con_esperado']}.",
        ]
        if payload.get("sequence_accuracy") is not None:
            lines.append(f"- Exactitud de secuencia: {payload['sequence_accuracy'] * 100:.2f} %.")
        if payload.get("digit_accuracy") is not None:
            lines.append(f"- Exactitud por dígito: {payload['digit_accuracy'] * 100:.2f} %.")
        lines += [
            "",
            "## Detalle",
            "",
            "| Archivo | Esperado | Predicho | Confianza | Correcto | Modelo |",
            "|---|---|---|---:|---|---|",
        ]
        for item in payload["resultados"]:
            lines.append(
                f"| {item['archivo']} | {item['esperado'] or '—'} | "
                f"{item['predicho'] or '—'} | {item['confianza'] * 100:.2f} % | "
                f"{'sí' if item['correcto'] else 'no'} | {item['modelo']} |"
            )
        lines += [
            "",
            "## Nota metodológica",
            "",
            "Las fotografías son aportadas por el usuario y se procesan localmente.",
            "El resultado depende de la iluminación, el encuadre, el contraste y la",
            "separación entre dígitos. Si una foto falla, se puede ajustar el área",
            "mínima o recortar la zona del número.",
            "",
        ]
    lines += [
        "## Archivos",
        "",
        f"- `{payload['json']}`",
        f"- `{payload['markdown']}`",
        f"- `{_relative(ANNOTATED_DIR)}`",
        "",
    ]
    REAL_RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    return REAL_RESULTS_MD


def run_real_photo_test(
    manifest_path: Path | None = None,
    model_name: str = "cnn",
    min_area: int = config.DEFAULT_MIN_AREA,
) -> dict[str, Any]:
    """Procesa las fotografías reales del manifiesto.

    Si no hay fotografías válidas, no inventa resultados: devuelve un estado
    `sin_fotografias` con instrucciones.
    """
    manifest = load_manifest(manifest_path)
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST

    if manifest.empty:
        payload = _empty_payload(
            manifest_file,
            mensaje=(
                "No hay fotografías reales registradas. Añade imágenes a "
                "data/real_photos y completa manifest.csv."
            ),
        )
        save_json(REAL_RESULTS_JSON, payload)
        _write_markdown(payload)
        return payload

    if not model_exists(model_name):
        raise FileNotFoundError(
            f"El modelo '{model_name}' no está entrenado. "
            f"Ejecuta: python main.py train --model {model_name}"
        )

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    sequence_hits = 0
    n_with_expected = 0
    digit_hits = 0
    digit_total = 0

    for index, row in manifest.iterrows():
        photo_type = str(row.get("tipo", "")).strip().lower()
        raw_name = str(row.get("archivo", "")).strip()
        if not raw_name:
            continue
        if photo_type != "foto_real":
            skipped.append(
                {
                    "archivo": raw_name,
                    "motivo": "tipo distinto de foto_real; no se cuenta como foto real",
                }
            )
            continue

        photo_path = _resolve_photo(row, manifest_file)
        if not photo_path.exists():
            skipped.append({"archivo": str(photo_path), "motivo": "archivo no encontrado"})
            continue

        row_model = str(row.get("modelo", "")).strip() or model_name
        if not model_exists(row_model):
            skipped.append(
                {"archivo": str(photo_path), "motivo": f"modelo no entrenado: {row_model}"}
            )
            continue
        row_min_area = row.get("min_area")
        if pd.isna(row_min_area) or str(row_min_area).strip() == "":
            row_min_area = min_area
        try:
            row_min_area = int(float(row_min_area))
        except (TypeError, ValueError):
            row_min_area = min_area

        model = load_trained_model(row_model)
        prediction = predict_image(
            photo_path,
            model,
            row_model,
            min_area=row_min_area,
        )

        expected = "" if pd.isna(row.get("esperado")) else str(row.get("esperado")).strip()
        predicted = prediction.number
        correct = (predicted == expected) if expected else False
        if expected:
            n_with_expected += 1
            if correct:
                sequence_hits += 1
            expected_digits = list(expected)
            if len(expected_digits) == len(prediction.digits):
                digit_total += len(expected_digits)
                digit_hits += sum(
                    int(str(pred) == exp)
                    for pred, exp in zip(prediction.digits, expected_digits)
                )

        annotated_name = f"{_safe_stem(photo_path.stem)}_{row_model}_anotada.png"
        annotated_path = ANNOTATED_DIR / annotated_name
        cv2.imwrite(str(annotated_path), prediction.annotated_image)

        results.append(
            {
                "archivo": _relative(photo_path),
                "esperado": expected,
                "predicho": predicted,
                "correcto": correct,
                "confianza": float(prediction.mean_confidence),
                "digitos": prediction.digits,
                "confianzas": [float(v) for v in prediction.confidences],
                "fallback": bool(prediction.fallback),
                "modelo": row_model,
                "min_area": row_min_area,
                "anotada": _relative(annotated_path),
                "descripcion": "" if pd.isna(row.get("descripcion")) else str(row.get("descripcion")),
            }
        )

    if not results:
        payload = _empty_payload(
            manifest_file,
            mensaje=(
                "El manifiesto no contiene fotografías reales válidas. "
                "Revisa que existan los archivos y que tipo=foto_real."
            ),
            skipped=skipped,
        )
        save_json(REAL_RESULTS_JSON, payload)
        _write_markdown(payload)
        return payload

    sequence_accuracy = (sequence_hits / n_with_expected) if n_with_expected else None
    digit_accuracy = (digit_hits / digit_total) if digit_total else None
    payload = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "estado": "procesado",
        "tipo_datos": "fotografias_reales",
        "nota": (
            "Solo se procesaron filas con tipo=foto_real. No se incluyen imágenes "
            "sintéticas de MNIST."
        ),
        "manifest": str(manifest_file),
        "modelo_principal": model_name,
        "n_fotos": len(results),
        "n_con_esperado": n_with_expected,
        "secuencias_correctas": sequence_hits,
        "sequence_accuracy": sequence_accuracy,
        "digit_accuracy": digit_accuracy,
        "resultados": results,
        "omitidos": skipped,
        "json": _relative(REAL_RESULTS_JSON),
        "markdown": _relative(REAL_RESULTS_MD),
    }
    save_json(REAL_RESULTS_JSON, payload)
    _write_markdown(payload)
    return payload


def _empty_payload(
    manifest_file: Path,
    mensaje: str,
    skipped: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Construye un resultado vacío honesto y con instrucciones."""
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "estado": "sin_fotografias",
        "tipo_datos": "fotografias_reales",
        "manifest": str(manifest_file),
        "n_fotos": 0,
        "n_con_esperado": 0,
        "secuencias_correctas": 0,
        "sequence_accuracy": None,
        "digit_accuracy": None,
        "resultados": [],
        "omitidos": skipped or [],
        "mensaje": mensaje,
        "instrucciones": (
            "Añade fotos reales a data/real_photos, completa manifest.csv con "
            "tipo=foto_real y ejecuta `python main.py real-test`."
        ),
        "json": _relative(REAL_RESULTS_JSON),
        "markdown": _relative(REAL_RESULTS_MD),
    }
