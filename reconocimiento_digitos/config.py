"""
Configuración central del proyecto.

Aquí se concentran rutas, constantes y hiperparámetros para evitar valores
"mágicos" repartidos por el código. Cualquier cambio de rutas o de tiempo de
entrenamiento debe hacerse en este archivo.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibilidad y avisos de TensorFlow
# ---------------------------------------------------------------------------
# Debe fijarse antes de importar TensorFlow para que los logs no ensucien la
# salida de consola. No oculta errores: solo mensajes informativos del runtime.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

SEED = 42
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
SAMPLES_DIR = DATA_DIR / "samples"
LABELS_DIR = DATA_DIR / "labels"
TESTS_DIR = BASE_DIR / "tests"

DATA_FILE = DATA_DIR / "mnist_split.npz"
DATASET_INFO_FILE = DATA_DIR / "dataset_info.json"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
IMAGE_SIZE = 28
NUM_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]
VALIDATION_SIZE = 6_000  # el conjunto de test oficial (10 000) nunca se toca

# ---------------------------------------------------------------------------
# Preprocesamiento de imágenes subidas por el usuario
# ---------------------------------------------------------------------------
MNIST_BOX_SIZE = 20       # caja estándar de MNIST: dígito de 20x20 en lienzo 28x28
MNIST_MARGIN = 4
DEFAULT_MIN_AREA = 40     # área mínima (px²) de un componente para considerarlo dígito
DEFAULT_MAX_DIGITS = 15

# ---------------------------------------------------------------------------
# Hiperparámetros de los modelos tradicionales
# ---------------------------------------------------------------------------
SVM_TRAIN_SIZE = 12_000   # subconjunto estratificado para que el SVM RBF sea viable
CV_SAMPLE_SIZE = 6_000    # subconjunto para validación cruzada (3 folds)
CV_FOLDS = 3
KNN_COMPONENTS = 50       # PCA previo: acelera KNN y reduce ruido
KNN_NEIGHBORS = 5
KMEANS_CLUSTERS = 10
KMEANS_SAMPLE_SIZE = 20_000
KMEANS_PCA_COMPONENTS = 50

# ---------------------------------------------------------------------------
# Hiperparámetros de Deep Learning
# ---------------------------------------------------------------------------
CNN_EPOCHS = 12
CNN_BATCH_SIZE = 128
CNN_LEARNING_RATE = 1e-3
MLP_EPOCHS = 25
MLP_BATCH_SIZE = 128
MLP_HIDDEN_UNITS = (256, 128)
MLP_LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 4

# Tamaño de lote para inferencia; no afecta el resultado.
PREDICT_BATCH_SIZE = 512

# ---------------------------------------------------------------------------
# Nombres legibles y artefactos
# ---------------------------------------------------------------------------
MODEL_DISPLAY_NAMES = {
    "logreg": "Regresión Logística",
    "kmeans": "K-Means (agrupamiento)",
    "knn": "K-Nearest Neighbors (KNN)",
    "svm": "Máquina de Vectores de Soporte (SVM)",
    "mlp": "Red Neuronal Artificial (MLP)",
    "cnn": "Red Neuronal Convolucional (CNN)",
}

SUPERVISED_MODELS = ["logreg", "knn", "svm", "mlp", "cnn"]

MODEL_FILES = {
    "logreg": MODELS_DIR / "logistic_regression.joblib",
    "kmeans": MODELS_DIR / "kmeans.joblib",
    "knn": MODELS_DIR / "knn.joblib",
    "svm": MODELS_DIR / "svm.joblib",
    "mlp": MODELS_DIR / "mlp.keras",
    "cnn": MODELS_DIR / "cnn.keras",
}

METRICS_FILE = RESULTS_DIR / "metrics.csv"
KMEANS_METRICS_FILE = RESULTS_DIR / "kmeans_metrics.json"
CV_RESULTS_FILE = RESULTS_DIR / "cross_validation.csv"
DEMO_RESULTS_FILE = RESULTS_DIR / "demo_predictions.json"
SUMMARY_FILE = RESULTS_DIR / "RESUMEN.md"
HISTORY_FILE = RESULTS_DIR / "historial_reconocimiento.json"


def ensure_directories() -> None:
    """Crea todas las carpetas de trabajo si no existen."""
    for directory in (
        BASE_DIR,
        DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        SAMPLES_DIR,
        LABELS_DIR,
        TESTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# Las carpetas se crean al importar el módulo: es una operación inocua y evita
# errores de "FileNotFoundError" en cualquier punto de entrada.
ensure_directories()
