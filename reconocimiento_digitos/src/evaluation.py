"""
Evaluación de modelos, generación de métricas y gráficos.

Principios:
- El conjunto de test (10 000 imágenes) se usa una sola vez para la evaluación
  final de cada modelo entrenado.
- La validación cruzada se aplica a los modelos tradicionales sobre un
  subconjunto estratificado, por viabilidad computacional. Sus resultados son
  estimaciones y se guardan aparte de las métricas de test.
- K-Means se evalúa como agrupamiento (ARI, NMI, silueta y pureza), nunca como
  clasificador supervisado. La "exactitud tras mapear clústeres" se incluye
  solo como interpretación a posteriori.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

import config
from src.preprocessing import (
    prepare_dataset_for_cnn,
    prepare_dataset_for_ml,
)
from src.training import load_trained_model, model_exists
from src.utils import save_json, stratified_subsample, timer

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)

CLASS_NAMES = config.CLASS_NAMES


# ---------------------------------------------------------------------------
# Predicción auxiliar
# ---------------------------------------------------------------------------
def predict_labels(model: Any, name: str, X: np.ndarray) -> np.ndarray:
    """Obtiene etiquetas predichas para un modelo Keras o scikit-learn."""
    if name in {"mlp", "cnn"}:
        probs = model.predict(X, batch_size=config.PREDICT_BATCH_SIZE, verbose=0)
        return np.argmax(probs, axis=1)
    return model.predict(X)


def predict_probabilities(model: Any, name: str, X: np.ndarray) -> np.ndarray:
    """Probabilidades (o puntuaciones tipo probabilidad) por clase."""
    if name in {"mlp", "cnn"}:
        return np.asarray(
            model.predict(X, batch_size=config.PREDICT_BATCH_SIZE, verbose=0),
            dtype=np.float32,
        )
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float32)

    # SVM sin calibración: softmax de los márgenes de decisión.
    scores = np.asarray(model.decision_function(X), dtype=np.float32)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def prepare_test_for(name: str, X_test: np.ndarray) -> np.ndarray:
    """Prepara el conjunto de test según el tipo de modelo."""
    if name == "cnn":
        return prepare_dataset_for_cnn(X_test)
    return prepare_dataset_for_ml(X_test)


# ---------------------------------------------------------------------------
# Métricas y matrices de confusión
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula accuracy, precision, recall y F1 (macro y ponderados)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
) -> Path:
    """Guarda la matriz de confusión en valores absolutos y normalizados."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=axes[0],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    axes[0].set_title("Recuentos")
    axes[0].set_xlabel("Dígito predicho")
    axes[0].set_ylabel("Dígito real")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=axes[1],
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Proporción por clase real")
    axes[1].set_xlabel("Dígito predicho")
    axes[1].set_ylabel("Dígito real")

    fig.suptitle(
        f"Matriz de confusión · {config.MODEL_DISPLAY_NAMES[name]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = config.PLOTS_DIR / f"confusion_matrix_{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_classifier(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evalúa un clasificador sobre test y guarda informe + matriz de confusión."""
    with timer() as t:
        y_pred = predict_labels(model, name, X_test)

    metrics = compute_metrics(y_test, y_pred)
    report_txt = classification_report(
        y_test,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        target_names=[f"Dígito {i}" for i in range(config.NUM_CLASSES)],
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(config.NUM_CLASSES)),
        output_dict=True,
        zero_division=0,
    )

    report_path = config.RESULTS_DIR / f"classification_report_{name}.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write(f"Informe de clasificación · {config.MODEL_DISPLAY_NAMES[name]}\n")
        fh.write("Conjunto: test oficial de MNIST (10 000 imágenes)\n")
        fh.write("=" * 68 + "\n")
        fh.write(report_txt)
        fh.write("\n")
        fh.write(f"Tiempo de inferencia: {t['seconds']:.3f} s\n")

    # Métricas por clase en CSV, útil para tablas detalladas.
    per_class_rows = []
    for digit in range(config.NUM_CLASSES):
        key = str(digit)
        row = report_dict.get(key, {})
        per_class_rows.append(
            {
                "digito": digit,
                "precision": row.get("precision", 0.0),
                "recall": row.get("recall", 0.0),
                "f1_score": row.get("f1-score", 0.0),
                "soporte": int(row.get("support", 0)),
            }
        )
    pd.DataFrame(per_class_rows).to_csv(
        config.RESULTS_DIR / f"classification_{name}.csv", index=False
    )

    plot_confusion_matrix(y_test, y_pred, name)

    row = {
        "modelo": name,
        "nombre": config.MODEL_DISPLAY_NAMES[name],
        **metrics,
        "tiempo_inferencia_s": t["seconds"],
        "n_test": int(len(y_test)),
    }
    return row


def load_metrics() -> pd.DataFrame:
    """Carga la tabla de métricas de test si existe."""
    if config.METRICS_FILE.exists():
        return pd.read_csv(config.METRICS_FILE)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Gráficos comparativos y curvas
# ---------------------------------------------------------------------------
def plot_model_comparison(metrics_df: pd.DataFrame) -> Path:
    """Gráfico de barras agrupadas comparando los clasificadores supervisados."""
    if metrics_df.empty:
        raise ValueError("No hay métricas para comparar.")

    df = metrics_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision (macro)"),
        ("recall_macro", "Recall (macro)"),
        ("f1_macro", "F1-score (macro)"),
    ]
    x = np.arange(len(df))
    width = 0.2
    colors = ["#2563eb", "#10b981", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ((column, label), color) in enumerate(zip(metrics, colors)):
        values = df[column].to_numpy() * 100
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["nombre"], rotation=12, ha="right")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Comparación de algoritmos sobre el conjunto de test")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = config.PLOTS_DIR / "model_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(name: str) -> Path | None:
    """Dibuja precisión y pérdida de entrenamiento/validación de una red."""
    path = config.RESULTS_DIR / f"history_{name}.csv"
    if not path.exists():
        return None

    history = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].plot(history["epoch"], history["accuracy"] * 100, marker="o", label="Entrenamiento")
    axes[0].plot(history["epoch"], history["val_accuracy"] * 100, marker="s", label="Validación")
    axes[0].set_title("Precisión por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["loss"], marker="o", label="Entrenamiento")
    axes[1].plot(history["epoch"], history["val_loss"], marker="s", label="Validación")
    axes[1].set_title("Pérdida por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Entropía cruzada")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Curvas de entrenamiento · {config.MODEL_DISPLAY_NAMES[name]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = config.PLOTS_DIR / f"training_{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sample_predictions(
    model: Any,
    name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n: int = 16,
) -> Path:
    """Muestra imágenes reales del test con su etiqueta real y la predicha."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    indices = rng.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    X_images = X_test[indices]
    y_true = y_test[indices]

    X_model = prepare_test_for(name, X_images)
    probs = predict_probabilities(model, name, X_model)
    y_pred = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)

    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, image, true, pred, conf in zip(axes, X_images, y_true, y_pred, confidence):
        ax.imshow(image, cmap="gray")
        color = "#15803d" if int(true) == int(pred) else "#b91c1c"
        ax.set_title(
            f"Real: {int(true)} · Pred: {int(pred)}\nConfianza: {conf * 100:.1f} %",
            color=color,
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"Predicciones sobre imágenes de test · {config.MODEL_DISPLAY_NAMES[name]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = config.PLOTS_DIR / f"sample_predictions_{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset_overview(splits: dict[str, np.ndarray]) -> None:
    """Genera un gráfico de distribución de clases y una rejilla de ejemplos."""
    counts = np.bincount(splits["y_train"], minlength=config.NUM_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    axes[0].bar(CLASS_NAMES, counts, color="#2563eb")
    axes[0].set_title("Distribución de clases en entrenamiento")
    axes[0].set_xlabel("Dígito")
    axes[0].set_ylabel("Imágenes")
    for i, value in enumerate(counts):
        axes[0].text(i, value + 30, str(int(value)), ha="center", fontsize=8)

    rng = np.random.default_rng(config.RANDOM_STATE)
    indices = rng.choice(len(splits["X_train"]), size=25, replace=False)
    grid = np.zeros((5 * 30, 5 * 30), dtype=np.uint8)
    for position, idx in enumerate(indices):
        row, col = divmod(position, 5)
        grid[row * 30 : row * 30 + 28, col * 30 : col * 30 + 28] = splits["X_train"][idx]
    axes[1].imshow(grid, cmap="gray")
    axes[1].set_title("Ejemplos del conjunto de entrenamiento")
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "dataset_overview.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# K-Means: evaluación como agrupamiento
# ---------------------------------------------------------------------------
def _cluster_mapping(cluster_labels: np.ndarray, y_true: np.ndarray) -> dict[int, int]:
    """Asigna a cada clúster el dígito mayoritario de sus miembros (a posteriori)."""
    mapping: dict[int, int] = {}
    for cluster in np.unique(cluster_labels):
        members = y_true[cluster_labels == cluster]
        if members.size:
            mapping[int(cluster)] = int(np.bincount(members, minlength=10).argmax())
    return mapping


def evaluate_kmeans(pipeline: Any, splits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Evalúa K-Means con métricas propias de agrupamiento.

    No se usa `accuracy` como métrica principal. Se calculan ARI, NMI,
    silueta y pureza. La exactitud tras mapear clústeres se reporta como
    interpretación, dejando claro que el algoritmo no fue entrenado con
    etiquetas.
    """
    X_train = prepare_dataset_for_ml(splits["X_train"])
    X_test = prepare_dataset_for_ml(splits["X_test"])
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    with timer() as t:
        train_clusters = pipeline.predict(X_train)
        test_clusters = pipeline.predict(X_test)

    mapping = _cluster_mapping(train_clusters, y_train)
    mapped_test = np.array([mapping.get(int(c), -1) for c in test_clusters])

    # Pureza: proporción de muestras que pertenecen al dígito mayoritario de su clúster.
    purity_hits = 0
    for cluster in np.unique(test_clusters):
        members = y_test[test_clusters == cluster]
        if members.size:
            purity_hits += int(np.bincount(members, minlength=10).max())
    purity = purity_hits / len(y_test)

    # Silueta sobre una muestra para limitar el coste cuadrático.
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_size = min(5000, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
    X_projected = pipeline.named_steps["pca"].transform(X_test[sample_idx])
    silhouette = float(silhouette_score(X_projected, test_clusters[sample_idx]))

    metrics = {
        "modelo": "kmeans",
        "nombre": config.MODEL_DISPLAY_NAMES["kmeans"],
        "tipo_evaluacion": "agrupamiento no supervisado",
        "n_clusters": int(config.KMEANS_CLUSTERS),
        "adjusted_rand_index": float(adjusted_rand_score(y_test, test_clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_test, test_clusters)),
        "silhouette": silhouette,
        "purity": float(purity),
        "exactitud_tras_mapeo": float(np.mean(mapped_test == y_test)),
        "inercia": float(pipeline.named_steps["kmeans"].inertia_),
        "tiempo_inferencia_s": t["seconds"],
        "mapeo_cluster_a_digito": {str(k): v for k, v in sorted(mapping.items())},
        "nota": (
            "K-Means no usa etiquetas. ARI, NMI y silueta evalúan la calidad del "
            "agrupamiento. La pureza y la exactitud tras mapeo se calculan a "
            "posteriori solo para interpretar los clústeres."
        ),
    }
    save_json(config.KMEANS_METRICS_FILE, metrics)
    _plot_kmeans(pipeline, splits, test_clusters, mapping, train_clusters)
    return metrics


def _plot_kmeans(
    pipeline: Any,
    splits: dict[str, np.ndarray],
    test_clusters: np.ndarray,
    mapping: dict[int, int],
    train_clusters: np.ndarray,
) -> None:
    """Genera centroides, dispersión PCA 2D, tamaños y matriz de clústeres."""
    pca = pipeline.named_steps["pca"]
    kmeans = pipeline.named_steps["kmeans"]
    X_test = prepare_dataset_for_ml(splits["X_test"])
    y_test = splits["y_test"]

    # 1. Centroides reconstruidos como imágenes.
    centers = pca.inverse_transform(kmeans.cluster_centers_).reshape(-1, 28, 28)
    sizes = np.bincount(train_clusters, minlength=config.KMEANS_CLUSTERS)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4))
    for cluster, ax in enumerate(axes.ravel()):
        ax.imshow(np.clip(centers[cluster], 0, 1), cmap="gray")
        ax.set_title(
            f"Clúster {cluster}\n→ dígito {mapping.get(cluster, '?')} · n={sizes[cluster]}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("K-Means · centroides reconstruidos y dígito mayoritario", fontsize=14)
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "kmeans_centers.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Dispersión PCA 2D: clústeres frente a etiquetas reales.
    rng = np.random.default_rng(config.RANDOM_STATE)
    sample_size = min(3000, len(X_test))
    idx = rng.choice(len(X_test), size=sample_size, replace=False)
    coords = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X_test[idx])
    clusters_sample = test_clusters[idx]
    labels_sample = y_test[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for cluster in range(config.KMEANS_CLUSTERS):
        mask = clusters_sample == cluster
        axes[0].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=7,
            alpha=0.65,
            label=f"C{cluster}",
        )
    axes[0].set_title("Agrupamiento encontrado por K-Means")
    axes[0].legend(markerscale=3, fontsize=7, ncol=2, frameon=False)
    axes[0].set_xlabel("Componente principal 1")
    axes[0].set_ylabel("Componente principal 2")

    for digit in range(config.NUM_CLASSES):
        mask = labels_sample == digit
        axes[1].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=7,
            alpha=0.65,
            label=str(digit),
        )
    axes[1].set_title("Etiquetas reales (referencia)")
    axes[1].legend(markerscale=3, fontsize=7, ncol=2, frameon=False)
    axes[1].set_xlabel("Componente principal 1")
    axes[1].set_ylabel("Componente principal 2")

    fig.suptitle("K-Means · proyección PCA de 3 000 imágenes de test", fontsize=14)
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "kmeans_clusters_pca.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Tamaño de los clústeres.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(config.KMEANS_CLUSTERS), sizes, color="#7c3aed")
    ax.set_xticks(range(config.KMEANS_CLUSTERS))
    ax.set_xlabel("Clúster")
    ax.set_ylabel("Número de imágenes de entrenamiento")
    ax.set_title("K-Means · tamaño de cada clúster")
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "kmeans_cluster_sizes.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Matriz de confusión del mapeo clúster -> dígito.
    mapped = np.array([mapping.get(int(c), -1) for c in test_clusters])
    cm = confusion_matrix(y_test, mapped, labels=list(range(config.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        square=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("K-Means · clústeres mapeados a dígitos (evaluación a posteriori)")
    ax.set_xlabel("Dígito asignado al clúster")
    ax.set_ylabel("Dígito real")
    plt.tight_layout()
    fig.savefig(config.PLOTS_DIR / "kmeans_confusion.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
def cross_validate_traditional_models(
    splits: dict[str, np.ndarray],
    models: tuple[str, ...] = ("logreg", "knn", "svm"),
) -> pd.DataFrame:
    """Validación cruzada estratificada de 3 folds sobre un subconjunto.

    Se usa un subconjunto de 6 000 imágenes (2 000 por fold de prueba) para
    que el coste sea razonable en CPU. El objetivo es comparar la estabilidad
    de los modelos tradicionales, no sustituir la evaluación final en test.
    """
    from src.models import build_model

    X = prepare_dataset_for_ml(splits["X_train"])
    y = splits["y_train"]
    X_sub, y_sub = stratified_subsample(
        X, y, config.CV_SAMPLE_SIZE, random_state=config.RANDOM_STATE
    )

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    rows = []
    for name in models:
        print(f"  Validación cruzada: {config.MODEL_DISPLAY_NAMES[name]}...")
        with timer() as t:
            scores = cross_validate(
                build_model(name),
                X_sub,
                y_sub,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
        row = {
            "modelo": name,
            "nombre": config.MODEL_DISPLAY_NAMES[name],
            "folds": config.CV_FOLDS,
            "n_muestras": len(X_sub),
            "tiempo_total_s": t["seconds"],
        }
        for metric in scoring:
            values = np.asarray(scores[f"test_{metric}"], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(config.CV_RESULTS_FILE, index=False)
    return df


# ---------------------------------------------------------------------------
# Orquestación de la evaluación
# ---------------------------------------------------------------------------
def evaluate_all_models(
    splits: dict[str, np.ndarray],
    models: list[str] | None = None,
    run_cv: bool = True,
) -> pd.DataFrame:
    """Evalúa todos los clasificadores entrenados y genera los gráficos."""
    models = models or config.SUPERVISED_MODELS
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    rows: list[dict[str, Any]] = []
    for name in models:
        if name not in config.MODEL_FILES or not model_exists(name):
            print(f"  [omitido] {name}: falta el modelo entrenado.")
            continue
        print(f"Evaluando {config.MODEL_DISPLAY_NAMES[name]}...")
        model = load_trained_model(name)
        X = prepare_test_for(name, X_test)
        row = evaluate_classifier(model, name, X, y_test)
        rows.append(row)
        print(
            f"  accuracy={row['accuracy'] * 100:.2f} % | "
            f"F1 macro={row['f1_macro'] * 100:.2f} %"
        )

    if not rows:
        raise RuntimeError(
            "No hay modelos entrenados que evaluar. Ejecuta primero "
            "`python main.py train --model all`."
        )

    metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    metrics_df.to_csv(config.METRICS_FILE, index=False)
    plot_model_comparison(metrics_df)

    for name in ("mlp", "cnn"):
        if model_exists(name):
            plot_training_curves(name)

    # Predicciones de ejemplo con el mejor modelo disponible.
    best_name = metrics_df.iloc[0]["modelo"]
    best_model = load_trained_model(best_name)
    plot_sample_predictions(best_model, best_name, X_test, y_test)

    # K-Means, si está entrenado.
    if model_exists("kmeans"):
        print("Evaluando K-Means como agrupamiento...")
        evaluate_kmeans(load_trained_model("kmeans"), splits)

    plot_dataset_overview(splits)

    if run_cv:
        print("Ejecutando validación cruzada de modelos tradicionales...")
        cross_validate_traditional_models(splits)

    return metrics_df


def get_metrics_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una tabla formateada para consola/README (porcentajes)."""
    if metrics_df.empty:
        return metrics_df
    table = metrics_df[
        [
            "nombre",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "tiempo_inferencia_s",
        ]
    ].copy()
    for column in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        table[column] = (table[column] * 100).round(2)
    table.columns = [
        "Modelo",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 macro (%)",
        "Inferencia (s)",
    ]
    return table
