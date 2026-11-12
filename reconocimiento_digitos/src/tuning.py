"""
Experimentos de ajuste de hiperparámetros para la CNN.

Este módulo fue añadido para el trabajo final. Entrena varias variantes de la
CNN sobre un subconjunto estratificado del conjunto de entrenamiento y evalúa
siempre con el conjunto de validación. El conjunto de test no se toca en ningún
momento durante estos experimentos.

El objetivo es comparar de forma real y reproducible:
- learning rate;
- cantidad de filtros por bloque convolucional;
- dropout;
- batch size.

Los resultados se guardan en results/cnn_tuning.csv, results/cnn_tuning.json,
results/TUNING_CNN.md, results/tuning/history_<id>.csv y plots/cnn_tuning.png.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.models import build_cnn, count_parameters
from src.preprocessing import prepare_dataset_for_cnn
from src.utils import save_json, timer

TUNING_DIR = config.RESULTS_DIR / "tuning"
TUNING_CSV = config.RESULTS_DIR / "cnn_tuning.csv"
TUNING_JSON = config.RESULTS_DIR / "cnn_tuning.json"
TUNING_MD = config.RESULTS_DIR / "TUNING_CNN.md"
TUNING_PLOT = config.PLOTS_DIR / "cnn_tuning.png"

# Cada experimento cambia un único factor respecto a la configuración base,
# salvo el experimento base. Así la comparación es interpretable.
EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "base",
        "factor": "configuración base",
        "descripcion": "LR 1e-3, filtros 32/64, dropout 0.25/0.50, batch 128",
        "learning_rate": 1e-3,
        "filters": (32, 64),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 128,
    },
    {
        "id": "lr_5e-4",
        "factor": "learning rate",
        "descripcion": "Learning rate más conservador: 5e-4",
        "learning_rate": 5e-4,
        "filters": (32, 64),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 128,
    },
    {
        "id": "lr_2e-3",
        "factor": "learning rate",
        "descripcion": "Learning rate más agresivo: 2e-3",
        "learning_rate": 2e-3,
        "filters": (32, 64),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 128,
    },
    {
        "id": "filtros_16_32",
        "factor": "cantidad de filtros",
        "descripcion": "CNN más ligera: filtros 16/32",
        "learning_rate": 1e-3,
        "filters": (16, 32),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 128,
    },
    {
        "id": "filtros_48_96",
        "factor": "cantidad de filtros",
        "descripcion": "CNN más capacidad: filtros 48/96",
        "learning_rate": 1e-3,
        "filters": (48, 96),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 128,
    },
    {
        "id": "dropout_bajo",
        "factor": "dropout",
        "descripcion": "Regularización más suave: dropout 0.15/0.35",
        "learning_rate": 1e-3,
        "filters": (32, 64),
        "dropout_conv": 0.15,
        "dropout_dense": 0.35,
        "batch_size": 128,
    },
    {
        "id": "dropout_alto",
        "factor": "dropout",
        "descripcion": "Regularización más fuerte: dropout 0.35/0.60",
        "learning_rate": 1e-3,
        "filters": (32, 64),
        "dropout_conv": 0.35,
        "dropout_dense": 0.60,
        "batch_size": 128,
    },
    {
        "id": "batch_64",
        "factor": "batch size",
        "descripcion": "Lotes más pequeños: batch 64",
        "learning_rate": 1e-3,
        "filters": (32, 64),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 64,
    },
    {
        "id": "batch_256",
        "factor": "batch size",
        "descripcion": "Lotes más grandes: batch 256",
        "learning_rate": 1e-3,
        "filters": (32, 64),
        "dropout_conv": 0.25,
        "dropout_dense": 0.50,
        "batch_size": 256,
    },
]


def _select_experiments(ids: Iterable[str] | None) -> list[dict[str, Any]]:
    """Filtra experimentos por identificador, conservando el orden definido."""
    if not ids:
        return list(EXPERIMENTS)
    wanted = {value.strip() for value in ids if value.strip()}
    selected = [exp for exp in EXPERIMENTS if exp["id"] in wanted]
    unknown = wanted.difference(exp["id"] for exp in EXPERIMENTS)
    if unknown:
        raise ValueError(
            f"Experimentos desconocidos: {', '.join(sorted(unknown))}. "
            f"Opciones: {', '.join(exp['id'] for exp in EXPERIMENTS)}"
        )
    if not selected:
        raise ValueError("No se seleccionó ningún experimento.")
    return selected


def _training_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    size: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Toma un subconjunto estratificado del train, sin tocar validación ni test."""
    if size <= 0 or size >= len(X_train):
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=int(size),
        stratify=y_train,
        random_state=random_state,
    )
    return X_sub, y_sub


def _history_rows(history: Any, experiment_id: str) -> pd.DataFrame:
    """Convierte el historial de Keras en un DataFrame con números de época."""
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df.insert(0, "experimento", experiment_id)
    return df


def _plot_results(df: pd.DataFrame, selected_id: str) -> Path:
    """Dibuja accuracy y val_loss de cada experimento."""
    ordered = df.sort_values("best_val_accuracy", ascending=False).reset_index(drop=True)
    labels = ordered["id"].tolist()
    colors = ["#16a34a" if value == selected_id else "#2563eb" for value in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].bar(labels, ordered["best_val_accuracy"] * 100, color=colors)
    axes[0].set_title("Mejor accuracy de validación")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(
        max(0, float(ordered["best_val_accuracy"].min() * 100) - 2),
        min(100, float(ordered["best_val_accuracy"].max() * 100) + 1),
    )
    axes[0].tick_params(axis="x", rotation=35)
    for i, value in enumerate(ordered["best_val_accuracy"] * 100):
        axes[0].text(i, value + 0.05, f"{value:.2f}", ha="center", fontsize=8)

    axes[1].bar(labels, ordered["val_loss_best_epoch"], color=colors)
    axes[1].set_title("Pérdida de validación en la mejor época")
    axes[1].set_ylabel("val_loss")
    axes[1].tick_params(axis="x", rotation=35)
    for i, value in enumerate(ordered["val_loss_best_epoch"]):
        axes[1].text(i, value + 0.002, f"{value:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "Ajuste de hiperparámetros de la CNN · selección por validación",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    TUNING_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TUNING_PLOT, bbox_inches="tight")
    plt.close(fig)
    return TUNING_PLOT


def _write_markdown(df: pd.DataFrame, payload: dict[str, Any]) -> Path:
    """Genera el informe Markdown de los experimentos."""
    ordered = df.sort_values("best_val_accuracy", ascending=False)
    lines = [
        "# Ajuste de hiperparámetros de la CNN",
        "",
        f"Generado: {payload['fecha']}",
        "",
        "> Experimentos reales ejecutados sobre un subconjunto estratificado del",
        "> conjunto de entrenamiento. La selección se hizo exclusivamente con el",
        "> conjunto de validación. El conjunto de test no se utilizó.",
        "",
        "## Protocolo",
        "",
        f"- Subconjunto de entrenamiento: {payload['n_train']:,} imágenes.",
        f"- Validación: {payload['n_val']:,} imágenes.",
        f"- Épocas por experimento: {payload['epochs']}.",
        f"- Semilla: {payload['semilla']}.",
        "- Criterio de selección: mayor `best_val_accuracy`; desempate por menor `val_loss_best_epoch`.",
        "",
        "## Resultados",
        "",
        "| Experimento | Factor | LR | Filtros | Dropout | Batch | Mejor val accuracy | val_loss | Época |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            "| {id} | {factor} | {lr:g} | {filters} | {dc:g}/{dd:g} | {batch} | "
            "{acc:.4f} | {loss:.4f} | {epoch} |".format(
                id=row["id"],
                factor=row["factor"],
                lr=row["learning_rate"],
                filters=row["filtros"],
                dc=row["dropout_conv"],
                dd=row["dropout_dense"],
                batch=row["batch_size"],
                acc=row["best_val_accuracy"],
                loss=row["val_loss_best_epoch"],
                epoch=row["best_epoch"],
            )
        )
    selected = payload["seleccionado"]
    lines += [
        "",
        "## Configuración seleccionada",
        "",
        f"- Experimento: `{selected['id']}`.",
        f"- Descripción: {selected['descripcion']}.",
        f"- Mejor accuracy de validación: {selected['best_val_accuracy'] * 100:.2f} %.",
        f"- Pérdida de validación en la mejor época: {selected['val_loss_best_epoch']:.4f}.",
        "",
        "## Justificación",
        "",
        payload["justificacion"],
        "",
        "## Limitaciones del experimento",
        "",
        "- El ajuste se hizo sobre un subconjunto de entrenamiento y un número",
        "  reducido de épocas para que fuera viable en CPU. Es un cribado",
        "  comparativo, no un entrenamiento final con las 54 000 imágenes.",
        "- El modelo CNN de producción conserva su entrenamiento original con",
        "  54 000 imágenes y 12 épocas. No se sobrescribió ningún artefacto.",
        "- Las diferencias pequeñas entre configuraciones pueden deberse a la",
        "  variabilidad de inicialización y al número limitado de épocas.",
        "",
        "## Archivos",
        "",
        "- `results/cnn_tuning.csv`",
        "- `results/cnn_tuning.json`",
        "- `results/tuning/history_<experimento>.csv`",
        "- `plots/cnn_tuning.png`",
        "",
    ]
    TUNING_MD.write_text("\n".join(lines), encoding="utf-8")
    return TUNING_MD


def run_cnn_experiments(
    splits: dict[str, np.ndarray],
    experiment_ids: Iterable[str] | None = None,
    epochs: int = 5,
    subset_size: int = 20_000,
    verbose: int = 2,
) -> dict[str, Any]:
    """Ejecuta los experimentos de CNN y guarda todos los resultados.

    Parameters
    ----------
    splits:
        Diccionario con X_train, y_train, X_val, y_val, X_test, y_test.
    experiment_ids:
        Identificadores a ejecutar; None ejecuta todos.
    epochs:
        Épocas fijas por experimento.
    subset_size:
        Tamaño del subconjunto estratificado de train. 0 usa todo el train.
    verbose:
        Verbosidad de Keras.

    Returns
    -------
    dict
        Resumen con la tabla, configuración seleccionada y rutas generadas.
    """
    from tensorflow import keras

    if epochs <= 0:
        raise ValueError("epochs debe ser positivo.")
    if subset_size < 0:
        raise ValueError("subset_size no puede ser negativo.")

    selected_experiments = _select_experiments(experiment_ids)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    X_train_raw, y_train = _training_subset(
        splits["X_train"],
        splits["y_train"],
        subset_size,
        random_state=config.RANDOM_STATE,
    )
    X_train = prepare_dataset_for_cnn(X_train_raw)
    X_val = prepare_dataset_for_cnn(splits["X_val"])
    y_val = splits["y_val"]

    print(
        f"Experimentos CNN: {len(selected_experiments)} | "
        f"train={len(X_train):,} | val={len(X_val):,} | épocas={epochs}"
    )
    print("El conjunto de test no se usa en estos experimentos.")

    rows: list[dict[str, Any]] = []
    for experiment in selected_experiments:
        exp_id = str(experiment["id"])
        print(
            f"\n[{exp_id}] {experiment['descripcion']} | "
            f"LR={experiment['learning_rate']:g} | "
            f"filtros={experiment['filters']} | "
            f"dropout={experiment['dropout_conv']}/{experiment['dropout_dense']} | "
            f"batch={experiment['batch_size']}"
        )

        keras.backend.clear_session()
        keras.utils.set_random_seed(config.RANDOM_STATE)
        model = build_cnn(
            filters=experiment["filters"],
            dropout_conv=experiment["dropout_conv"],
            dropout_dense=experiment["dropout_dense"],
            learning_rate=experiment["learning_rate"],
        )

        with timer() as t:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=int(epochs),
                batch_size=int(experiment["batch_size"]),
                shuffle=True,
                verbose=verbose,
            )

        history_df = _history_rows(history, exp_id)
        history_path = TUNING_DIR / f"history_{exp_id}.csv"
        history_df.to_csv(history_path, index=False)

        val_acc = history_df["val_accuracy"].to_numpy(dtype=float)
        val_loss = history_df["val_loss"].to_numpy(dtype=float)
        best_index = int(np.argmax(val_acc))
        best_epoch = int(history_df.loc[best_index, "epoch"])

        row = {
            "id": exp_id,
            "factor": experiment["factor"],
            "descripcion": experiment["descripcion"],
            "learning_rate": float(experiment["learning_rate"]),
            "filters": tuple(int(v) for v in experiment["filters"]),
            "filtros": f"{experiment['filters'][0]}/{experiment['filters'][1]}",
            "dropout_conv": float(experiment["dropout_conv"]),
            "dropout_dense": float(experiment["dropout_dense"]),
            "batch_size": int(experiment["batch_size"]),
            "epochs_config": int(epochs),
            "epochs_ejecutadas": int(len(history_df)),
            "best_epoch": best_epoch,
            "best_val_accuracy": float(val_acc[best_index]),
            "val_loss_best_epoch": float(val_loss[best_index]),
            "final_val_accuracy": float(val_acc[-1]),
            "final_val_loss": float(val_loss[-1]),
            "final_train_accuracy": float(history_df["accuracy"].iloc[-1]),
            "final_train_loss": float(history_df["loss"].iloc[-1]),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "parametros_entrenables": int(count_parameters(model)),
            "tiempo_entrenamiento_s": float(t["seconds"]),
            "usando_test": False,
            "historial": str(history_path.relative_to(config.BASE_DIR)),
        }
        rows.append(row)
        print(
            f"  mejor val_accuracy={row['best_val_accuracy'] * 100:.2f} % | "
            f"val_loss={row['val_loss_best_epoch']:.4f} | "
            f"época={best_epoch} | tiempo={row['tiempo_entrenamiento_s']:.1f} s"
        )

        del model
        keras.backend.clear_session()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["best_val_accuracy", "val_loss_best_epoch"],
        ascending=[False, True],
    ).reset_index(drop=True)
    df.to_csv(TUNING_CSV, index=False)

    best = df.iloc[0].to_dict()
    baseline = df[df["id"] == "base"]
    baseline_acc = (
        float(baseline.iloc[0]["best_val_accuracy"]) if not baseline.empty else None
    )
    improve = (
        (float(best["best_val_accuracy"]) - baseline_acc) * 100
        if baseline_acc is not None
        else None
    )

    if best["id"] == "base":
        justificacion = (
            "La configuración base fue la mejor en validación. Por tanto, se "
            "mantiene la CNN de producción ya entrenada, sin sobrescribir su "
            "artefacto ni sus métricas de test. Los experimentos confirman que "
            "aumentar o reducir la capacidad, el dropout, el learning rate o el "
            "batch size no mejoró la validación en este protocolo."
        )
        accion = "mantener_cnn_actual"
    else:
        justificacion = (
            f"La configuración '{best['id']}' obtuvo la mejor accuracy de "
            f"validación ({best['best_val_accuracy'] * 100:.2f} %), con una "
            f"mejora de {improve:.2f} puntos porcentuales frente a la base. "
            "Se registra como configuración candidata. El modelo de producción "
            "no se sobrescribió: para adoptarla habría que reentrenar con las "
            "54 000 imágenes, reevaluar en test y actualizar los artefactos."
        )
        accion = "configuracion_candidata"

    payload = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "protocolo": "subconjunto estratificado de train; validación completa; test no usado",
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "epochs": int(epochs),
        "semilla": config.RANDOM_STATE,
        "experimentos": rows,
        "seleccionado": {
            "id": str(best["id"]),
            "descripcion": str(best["descripcion"]),
            "best_val_accuracy": float(best["best_val_accuracy"]),
            "val_loss_best_epoch": float(best["val_loss_best_epoch"]),
            "best_epoch": int(best["best_epoch"]),
            "learning_rate": float(best["learning_rate"]),
            "filtros": str(best["filtros"]),
            "dropout_conv": float(best["dropout_conv"]),
            "dropout_dense": float(best["dropout_dense"]),
            "batch_size": int(best["batch_size"]),
        },
        "mejora_vs_base_puntos": improve,
        "accion": accion,
        "justificacion": justificacion,
        "usando_test": False,
        "archivos": {
            "csv": str(TUNING_CSV.relative_to(config.BASE_DIR)),
            "json": str(TUNING_JSON.relative_to(config.BASE_DIR)),
            "markdown": str(TUNING_MD.relative_to(config.BASE_DIR)),
            "plot": str(TUNING_PLOT.relative_to(config.BASE_DIR)),
            "historiales": str(TUNING_DIR.relative_to(config.BASE_DIR)),
        },
    }
    save_json(TUNING_JSON, payload)
    _plot_results(df, str(best["id"]))
    _write_markdown(df, payload)

    print("\nResultados de ajuste de hiperparámetros:")
    print(
        df[
            [
                "id",
                "factor",
                "learning_rate",
                "filtros",
                "dropout_conv",
                "dropout_dense",
                "batch_size",
                "best_val_accuracy",
                "val_loss_best_epoch",
                "best_epoch",
            ]
        ].to_string(index=False)
    )
    print(f"\nConfiguración seleccionada: {best['id']}")
    print(justificacion)
    print(f"\nResultados: {TUNING_CSV}")
    print(f"Informe: {TUNING_MD}")
    print(f"Gráfico: {TUNING_PLOT}")

    return payload


def load_tuning_results() -> dict[str, Any]:
    """Carga el JSON de experimentos si existe; devuelve {} si no."""
    from src.utils import load_json

    if not TUNING_JSON.exists():
        return {}
    return load_json(TUNING_JSON)
