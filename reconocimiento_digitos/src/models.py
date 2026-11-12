"""
Definición y construcción de todos los modelos del proyecto.

Modelos supervisados:
- Regresión Logística (scikit-learn)
- KNN con reducción de dimensionalidad PCA (scikit-learn)
- SVM con kernel RBF (scikit-learn)
- Red neuronal artificial MLP (TensorFlow/Keras)
- Red neuronal convolucional CNN (TensorFlow/Keras)

Modelo no supervisado:
- K-Means (scikit-learn), que agrupa imágenes sin usar etiquetas. No se
  presenta como clasificador: sus métricas son de agrupamiento (ARI, NMI,
  silueta) y la "pureza" se calcula a posteriori para interpretar los clústeres.
"""

from __future__ import annotations

from typing import Any

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import config


# ---------------------------------------------------------------------------
# Modelos clásicos
# ---------------------------------------------------------------------------
def build_logistic_regression() -> LogisticRegression:
    """Regresión logística multinomial sobre los 784 píxeles normalizados."""
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=400,
        tol=1e-3,
        random_state=config.RANDOM_STATE,
    )


def build_knn() -> Pipeline:
    """KNN sobre 50 componentes principales.

    PCA reduce ruido y acelera la búsqueda de vecinos. Con 784 dimensiones el
    cálculo de distancias sería mucho más costoso y sufriría la maldición de la
    dimensionalidad.
    """
    return Pipeline(
        steps=[
            ("pca", PCA(n_components=config.KNN_COMPONENTS, random_state=config.RANDOM_STATE)),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=config.KNN_NEIGHBORS,
                    weights="distance",
                    metric="euclidean",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_svm() -> SVC:
    """SVM con kernel RBF.

    No se activa la calibración de probabilidades (`probability=True`) porque
    multiplica el coste de entrenamiento. La confianza mostrada en la
    aplicación se obtiene aplicando softmax a los márgenes de decisión
    (ver `src.prediction`).
    """
    return SVC(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        cache_size=1000,
        random_state=config.RANDOM_STATE,
    )


def build_kmeans() -> Pipeline:
    """K-Means sobre 50 componentes principales, con 10 clústeres.

    Se usa un Pipeline PCA + KMeans por eficiencia. Los centroides se pueden
    reconstruir al espacio de 784 píxeles con `inverse_transform` para
    visualizarlos como imágenes.
    """
    return Pipeline(
        steps=[
            (
                "pca",
                PCA(
                    n_components=config.KMEANS_PCA_COMPONENTS,
                    random_state=config.RANDOM_STATE,
                ),
            ),
            (
                "kmeans",
                KMeans(
                    n_clusters=config.KMEANS_CLUSTERS,
                    init="k-means++",
                    n_init=10,
                    max_iter=300,
                    tol=1e-4,
                    random_state=config.RANDOM_STATE,
                    algorithm="lloyd",
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Redes neuronales (Keras)
# ---------------------------------------------------------------------------
def build_mlp() -> Any:
    """Red neuronal densa (MLP) para compararla con los modelos clásicos."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(config.IMAGE_SIZE * config.IMAGE_SIZE,), name="pixeles"),
            layers.Dense(config.MLP_HIDDEN_UNITS[0], activation="relu", name="oculta_1"),
            layers.Dropout(0.20, name="dropout_1"),
            layers.Dense(config.MLP_HIDDEN_UNITS[1], activation="relu", name="oculta_2"),
            layers.Dropout(0.20, name="dropout_2"),
            layers.Dense(config.NUM_CLASSES, activation="softmax", name="salida"),
        ],
        name="mlp_mnist",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.MLP_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn() -> Any:
    """CNN con dos bloques convolucionales, MaxPooling, Flatten y Dense.

    Arquitectura:
    Conv2D(32) -> BatchNorm -> Conv2D(32) -> MaxPool -> Dropout
    Conv2D(64) -> BatchNorm -> Conv2D(64) -> MaxPool -> Dropout
    Flatten -> Dense(128, ReLU) -> Dropout -> Dense(10, Softmax)

    Se usa ReLU en las capas ocultas, Softmax en la salida, Adam como
    optimizador y entropía cruzada categórica dispersa como pérdida.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 1), name="imagen"),
            # Bloque convolucional 1
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv_1"),
            layers.BatchNormalization(name="bn_1"),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv_2"),
            layers.MaxPooling2D((2, 2), name="pool_1"),
            layers.Dropout(0.25, name="dropout_conv_1"),
            # Bloque convolucional 2
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_3"),
            layers.BatchNormalization(name="bn_2"),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_4"),
            layers.MaxPooling2D((2, 2), name="pool_2"),
            layers.Dropout(0.25, name="dropout_conv_2"),
            # Clasificador
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="densa"),
            layers.Dropout(0.50, name="dropout_densa"),
            layers.Dense(config.NUM_CLASSES, activation="softmax", name="salida"),
        ],
        name="cnn_mnist",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def build_model(name: str) -> Any:
    """Devuelve un modelo sin entrenar a partir de su clave corta."""
    builders = {
        "logreg": build_logistic_regression,
        "knn": build_knn,
        "svm": build_svm,
        "kmeans": build_kmeans,
        "mlp": build_mlp,
        "cnn": build_cnn,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(
            f"Modelo desconocido: {name!r}. Opciones: {sorted(builders)}"
        ) from exc


def is_keras_model(model: Any) -> bool:
    """Detecta si un modelo es de Keras sin importar TensorFlow al inicio."""
    return hasattr(model, "predict") and hasattr(model, "fit") and hasattr(model, "layers")


def count_parameters(model: Any) -> int:
    """Número de parámetros entrenables de un modelo Keras."""
    try:
        return int(sum(int(p.numpy().size) for p in model.trainable_weights))
    except Exception:
        return 0


def model_summary(name: str) -> str:
    """Resumen textual de la arquitectura para consola y resultados."""
    if name == "cnn":
        return (
            "CNN: Conv2D(32)-BN-Conv2D(32)-MaxPool-Dropout | "
            "Conv2D(64)-BN-Conv2D(64)-MaxPool-Dropout | Flatten-Dense(128)-Dropout-Dense(10)"
        )
    if name == "mlp":
        return (
            "MLP: Dense(256)-Dropout-Dense(128)-Dropout-Dense(10), "
            "ReLU + Softmax, Adam, sparse_categorical_crossentropy"
        )
    if name == "svm":
        return "SVM RBF (C=10, gamma=scale) sobre 784 píxeles; subconjunto estratificado"
    if name == "knn":
        return f"KNN (k={config.KNN_NEIGHBORS}, pesos=distancia) sobre PCA({config.KNN_COMPONENTS})"
    if name == "logreg":
        return "Regresión logística multinomial (lbfgs, C=1.0)"
    if name == "kmeans":
        return f"K-Means ({config.KMEANS_CLUSTERS} clústeres) sobre PCA({config.KMEANS_PCA_COMPONENTS})"
    return name
