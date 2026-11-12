"""
Entrenamiento de los modelos y guardado de artefactos.

Cada modelo entrenado se guarda en `models/` y sus metadatos en
`results/training_<modelo>.json`. Los modelos de Keras guardan además su
historial de entrenamiento en `results/history_<modelo>.csv`, que luego se usa
para dibujar las curvas de precisión y pérdida.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import config
from src.models import (
    build_cnn,
    build_kmeans,
    build_knn,
    build_logistic_regression,
    build_mlp,
    build_svm,
    count_parameters,
    model_summary,
)
from src.preprocessing import prepare_dataset_for_cnn, prepare_dataset_for_ml
from src.utils import save_json, stratified_subsample, timer

TRAIN_ORDER = ["logreg", "knn", "svm", "kmeans", "mlp", "cnn"]


def history_path(name: str) -> Path:
    """Ruta del CSV con el historial de entrenamiento de una red."""
    return config.RESULTS_DIR / f"history_{name}.csv"


def training_metadata_path(name: str) -> Path:
    """Ruta del JSON con metadatos del entrenamiento."""
    return config.RESULTS_DIR / f"training_{name}.json"


def model_exists(name: str) -> bool:
    """Comprueba si el artefacto del modelo existe en disco."""
    return config.MODEL_FILES[name].exists()


def load_trained_model(name: str) -> Any:
    """Carga un modelo entrenado desde `models/`.

    Raises
    ------
    FileNotFoundError
        Si el modelo aún no se ha entrenado.
    """
    path = config.MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo '{name}' en {path}. "
            f"Ejecuta: python main.py train --model {name}"
        )
    if name in {"mlp", "cnn"}:
        from tensorflow import keras

        return keras.models.load_model(path, compile=False)
    return joblib.load(path)


def _save_classic_model(name: str, model: Any) -> Path:
    path = config.MODEL_FILES[name]
    joblib.dump(model, path, compress=3)
    return path


def _predict_labels(model: Any, X: np.ndarray, name: str) -> np.ndarray:
    """Predicciones para cualquier modelo (Keras o scikit-learn)."""
    if name in {"mlp", "cnn"}:
        probs = model.predict(X, batch_size=config.PREDICT_BATCH_SIZE, verbose=0)
        return np.argmax(probs, axis=1)
    return model.predict(X)


def _val_accuracy(name: str, model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Exactitud sobre validación, solo informativa para modelos clásicos."""
    if name in {"mlp", "cnn"}:
        X = prepare_dataset_for_ml(X_val) if name == "mlp" else prepare_dataset_for_cnn(X_val)
    else:
        X = prepare_dataset_for_ml(X_val)
    y_pred = _predict_labels(model, X, name)
    return float(np.mean(y_pred == y_val))


def _base_metadata(name: str, n_train: int) -> dict[str, Any]:
    return {
        "modelo": name,
        "nombre_legible": config.MODEL_DISPLAY_NAMES[name],
        "arquitectura": model_summary(name),
        "n_entrenamiento": int(n_train),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "semilla": config.RANDOM_STATE,
    }


# ---------------------------------------------------------------------------
# Modelos clásicos
# ---------------------------------------------------------------------------
def train_classic_model(name: str, splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Entrena Regresión Logística, KNN o SVM y guarda el modelo.

    El SVM se entrena con un subconjunto estratificado de 12 000 imágenes para
    que el kernel RBF sea viable en CPU; el resto de modelos usan las 54 000.
    """
    X_train = prepare_dataset_for_ml(splits["X_train"])
    y_train = splits["y_train"]
    X_val = prepare_dataset_for_ml(splits["X_val"])

    if name == "svm":
        X_fit, y_fit = stratified_subsample(
            X_train,
            y_train,
            config.SVM_TRAIN_SIZE,
            random_state=config.RANDOM_STATE,
        )
    else:
        X_fit, y_fit = X_train, y_train

    builders = {
        "logreg": build_logistic_regression,
        "knn": build_knn,
        "svm": build_svm,
    }
    model = builders[name]()

    print(f"\nEntrenando {config.MODEL_DISPLAY_NAMES[name]} con {len(X_fit):,} imágenes...")
    with timer() as t:
        model.fit(X_fit, y_fit)

    val_acc = _val_accuracy(name, model, X_val, splits["y_val"])
    path = _save_classic_model(name, model)

    metadata = _base_metadata(name, len(X_fit))
    metadata.update(
        {
            "tiempo_entrenamiento_s": t["seconds"],
            "exactitud_validacion": round(val_acc, 6),
            "archivo": str(path.relative_to(config.BASE_DIR)),
            "nota": (
                "Entrenado sobre un subconjunto estratificado por coste computacional."
                if name == "svm"
                else "Entrenado con el conjunto de entrenamiento completo."
            ),
        }
    )
    save_json(training_metadata_path(name), metadata)
    print(f"  Validación: {val_acc * 100:.2f} % | tiempo: {t['seconds']:.1f} s")
    return metadata


# ---------------------------------------------------------------------------
# K-Means (no supervisado)
# ---------------------------------------------------------------------------
def train_kmeans(splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Entrena K-Means sin utilizar etiquetas.

    Se toma una muestra aleatoria del conjunto de entrenamiento. El muestreo
    no es estratificado a propósito: K-Means no debe ver las etiquetas ni
    siquiera para construir la muestra.
    """
    X_train = prepare_dataset_for_ml(splits["X_train"])
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = min(config.KMEANS_SAMPLE_SIZE, len(X_train))
    indices = rng.choice(len(X_train), size=n, replace=False)
    X_fit = X_train[indices]

    model = build_kmeans()
    print(f"\nEntrenando K-Means (no supervisado) con {n:,} imágenes...")
    with timer() as t:
        model.fit(X_fit)

    path = _save_classic_model("kmeans", model)
    metadata = _base_metadata("kmeans", n)
    metadata.update(
        {
            "tiempo_entrenamiento_s": t["seconds"],
            "inercia": float(model.named_steps["kmeans"].inertia_),
            "archivo": str(path.relative_to(config.BASE_DIR)),
            "nota": (
                "Modelo de agrupamiento: no usa etiquetas durante el ajuste. "
                "La asignación de dígitos se hace a posteriori solo para interpretar."
            ),
        }
    )
    save_json(training_metadata_path("kmeans"), metadata)
    print(f"  Inercia: {metadata['inercia']:.1f} | tiempo: {t['seconds']:.1f} s")
    return metadata


# ---------------------------------------------------------------------------
# Redes neuronales
# ---------------------------------------------------------------------------
def train_keras_model(name: str, splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Entrena la MLP o la CNN con validación y callbacks."""
    from tensorflow import keras

    if name == "mlp":
        X_train = prepare_dataset_for_ml(splits["X_train"])
        X_val = prepare_dataset_for_ml(splits["X_val"])
        model = build_mlp()
        epochs = config.MLP_EPOCHS
        batch_size = config.MLP_BATCH_SIZE
    elif name == "cnn":
        X_train = prepare_dataset_for_cnn(splits["X_train"])
        X_val = prepare_dataset_for_cnn(splits["X_val"])
        model = build_cnn()
        epochs = config.CNN_EPOCHS
        batch_size = config.CNN_BATCH_SIZE
    else:
        raise ValueError(f"Modelo Keras no soportado: {name}")

    y_train = splits["y_train"]
    y_val = splits["y_val"]

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(config.MODEL_FILES[name]),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    )
    callbacks = [checkpoint, early_stopping, reduce_lr]

    print(
        f"\nEntrenando {config.MODEL_DISPLAY_NAMES[name]} "
        f"({epochs} épocas máx., lote {batch_size})..."
    )
    print(model.summary())

    with timer() as t:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )

    # Guardar historial completo para las curvas.
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(history_path(name), index=False)

    # EarlyStopping restauró en memoria los pesos de la mejor época. Se guarda
    # ese modelo directamente: ya está compilado y no hace falta recargarlo
    # (recargar con compile=False impediría llamar a evaluate()).
    model.save(config.MODEL_FILES[name])

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    best_epoch = int(np.argmax(hist_df["val_accuracy"].to_numpy()) + 1)

    metadata = _base_metadata(name, len(X_train))
    metadata.update(
        {
            "tiempo_entrenamiento_s": t["seconds"],
            "epocas_ejecutadas": int(len(hist_df)),
            "mejor_epoca": best_epoch,
            "mejor_val_accuracy": float(np.max(hist_df["val_accuracy"])),
            "val_loss": float(val_loss),
            "exactitud_validacion": float(val_acc),
            "parametros_entrenables": count_parameters(model),
            "archivo": str(config.MODEL_FILES[name].relative_to(config.BASE_DIR)),
            "historial": str(history_path(name).relative_to(config.BASE_DIR)),
            "nota": "Validación usada para early stopping y selección del mejor modelo.",
        }
    )
    save_json(training_metadata_path(name), metadata)
    print(
        f"  Mejor validación: {metadata['mejor_val_accuracy'] * 100:.2f} % "
        f"(época {best_epoch}) | tiempo: {t['seconds']:.1f} s"
    )
    return metadata


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def train_model(name: str, splits: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Entrena un modelo concreto y devuelve sus metadatos."""
    if splits is None:
        from src.dataset import load_dataset

        splits = load_dataset()

    if name in {"logreg", "knn", "svm"}:
        return train_classic_model(name, splits)
    if name == "kmeans":
        return train_kmeans(splits)
    if name in {"mlp", "cnn"}:
        return train_keras_model(name, splits)
    raise ValueError(f"Modelo no soportado: {name}")


def train_all(
    models: list[str] | None = None,
    splits: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Entrena varios modelos, continuando si alguno falla.

    Returns
    -------
    (metadatos, errores)
        Lista de metadatos de los modelos entrenados y lista de pares
        (modelo, mensaje_de_error).
    """
    models = models or TRAIN_ORDER
    if splits is None:
        from src.dataset import load_dataset

        splits = load_dataset()

    metadata: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []

    for name in models:
        if name not in config.MODEL_FILES:
            errors.append((name, "Modelo desconocido"))
            continue
        try:
            metadata.append(train_model(name, splits))
        except Exception as exc:  # se informa y se sigue con el resto
            errors.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"\n[ERROR] No se pudo entrenar '{name}': {exc}")

    return metadata, errors
