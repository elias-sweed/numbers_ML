"""
Punto de entrada de línea de comandos del proyecto.

Ejemplos:
    python main.py prepare
    python main.py train --model all
    python main.py evaluate
    python main.py tune-cnn
    python main.py demo
    python main.py real-test
    python main.py all
    python main.py predict --image ruta/a/foto.png --model cnn
    python main.py info

El proyecto también puede usarse desde la interfaz Streamlit (`app.py`).
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

# Garantiza que `import src...` funcione aunque el script se invoque desde otra
# carpeta con una ruta absoluta.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src.dataset import describe_dataset, load_dataset, load_dataset_info, prepare_dataset  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_all_models,
    get_metrics_table,
    load_metrics,
)
from src.training import TRAIN_ORDER, load_trained_model, model_exists, train_all  # noqa: E402
from src.utils import format_seconds, human_bytes  # noqa: E402


# ---------------------------------------------------------------------------
# Utilidades de presentación
# ---------------------------------------------------------------------------
def header(title: str) -> None:
    """Imprime un encabezado legible en consola."""
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "no instalado"


def _model_size(name: str) -> str:
    path = config.MODEL_FILES[name]
    if not path.exists():
        return "—"
    return human_bytes(path.stat().st_size)


def _parse_models(values: list[str] | None) -> list[str] | None:
    """Convierte ['all'] o ['cnn,mlp'] en una lista de claves válidas."""
    if not values:
        return None
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                return list(TRAIN_ORDER)
            if part not in config.MODEL_FILES:
                raise ValueError(
                    f"Modelo desconocido: {part!r}. Opciones: all, {', '.join(TRAIN_ORDER)}"
                )
            result.append(part)
    return list(dict.fromkeys(result))


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------
def command_prepare(args: argparse.Namespace) -> None:
    header("FASE 2 · Preparación del dataset MNIST")
    splits = prepare_dataset(force=args.force)
    print(describe_dataset(splits).to_string(index=False))
    info = load_dataset_info()
    print(f"\nTotal de imágenes: {info['conteos']['total']:,}")
    print("Dimensiones: 28 x 28 píxeles, 1 canal (escala de grises)")
    print(f"Entrenamiento: {info['conteos']['train']:,}")
    print(f"Validación:    {info['conteos']['validation']:,}")
    print(f"Prueba:        {info['conteos']['test']:,} (intacto para evaluación final)")
    print(f"\nEtiquetas guardadas en: {config.LABELS_DIR}")
    print(f"Metadatos: {config.DATASET_INFO_FILE}")


def command_train(args: argparse.Namespace) -> None:
    header("FASE 4-5 · Entrenamiento de modelos")
    splits = load_dataset()
    names = _parse_models(args.model) or list(TRAIN_ORDER)

    if args.skip_existing:
        skipped = [name for name in names if model_exists(name)]
        names = [name for name in names if not model_exists(name)]
        if skipped:
            print(f"Modelos ya entrenados (omitidos): {', '.join(skipped)}")
    if not names:
        print("No hay modelos pendientes de entrenar.")
        return

    print(f"Se entrenarán: {', '.join(config.MODEL_DISPLAY_NAMES[n] for n in names)}")
    metadata, errors = train_all(names, splits)

    header("Resumen del entrenamiento")
    rows = []
    for meta in metadata:
        rows.append(
            {
                "Modelo": meta["nombre_legible"],
                "Muestras": f"{meta['n_entrenamiento']:,}",
                "Val accuracy (%)": round(meta.get("exactitud_validacion", 0) * 100, 2),
                "Tiempo": format_seconds(meta["tiempo_entrenamiento_s"]),
                "Archivo": Path(meta["archivo"]).name,
                "Tamaño": _model_size(meta["modelo"]),
            }
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    if errors:
        print("\nErrores:")
        for name, message in errors:
            print(f"  - {name}: {message}")
        raise SystemExit(1)


def command_evaluate(args: argparse.Namespace) -> None:
    header("FASE 6-7 · Evaluación, métricas y gráficos")
    splits = load_dataset()
    names = _parse_models(args.models)
    metrics_df = evaluate_all_models(splits, names, run_cv=not args.no_cv)

    print("\nResultados sobre el conjunto de test:")
    print(get_metrics_table(metrics_df).to_string(index=False))

    if config.KMEANS_METRICS_FILE.exists():
        from src.utils import load_json

        km = load_json(config.KMEANS_METRICS_FILE)
        print("\nK-Means (evaluación de agrupamiento, no supervisado):")
        print(f"  ARI: {km['adjusted_rand_index']:.4f}")
        print(f"  NMI: {km['normalized_mutual_info']:.4f}")
        print(f"  Silueta: {km['silhouette']:.4f}")
        print(f"  Pureza: {km['purity'] * 100:.2f} %")
        print(f"  Exactitud tras mapeo (solo interpretativa): {km['exactitud_tras_mapeo'] * 100:.2f} %")

    if config.CV_RESULTS_FILE.exists():
        cv = pd.read_csv(config.CV_RESULTS_FILE)
        print("\nValidación cruzada (3 folds, subconjunto estratificado):")
        for _, row in cv.iterrows():
            print(
                f"  {row['nombre']}: accuracy {row['accuracy_mean'] * 100:.2f} % "
                f"± {row['accuracy_std'] * 100:.2f} %"
            )

    print(f"\nTabla de métricas: {config.METRICS_FILE}")
    print(f"Gráficos: {config.PLOTS_DIR}")
    write_summary()


def command_demo(args: argparse.Namespace) -> None:
    header("FASE 11 · Prueba del pipeline con imágenes de demostración")
    from src.demo import generate_demo_report

    payload = generate_demo_report(args.model)
    if payload["results"]:
        rows = [
            {
                "Archivo": item["file"],
                "Esperado": item["expected"],
                "Predicho": item["predicted"],
                "Confianza (%)": round(item["confidence"] * 100, 2),
                "Correcto": "sí" if item["correct"] else "no",
            }
            for item in payload["results"]
        ]
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nResultados: {config.DEMO_RESULTS_FILE}")


def command_tune_cnn(args: argparse.Namespace) -> None:
    header("FASE 5 · Ajuste de hiperparámetros de la CNN")
    from src.tuning import run_cnn_experiments

    splits = load_dataset()
    payload = run_cnn_experiments(
        splits,
        experiment_ids=args.experiments,
        epochs=args.epochs,
        subset_size=args.subset,
        verbose=0 if args.quiet else 2,
    )
    print("\nConfiguración seleccionada por validación:")
    print(f"  {payload['seleccionado']['id']}: {payload['seleccionado']['descripcion']}")
    print(f"  Accuracy de validación: {payload['seleccionado']['best_val_accuracy'] * 100:.2f} %")
    print(f"  val_loss: {payload['seleccionado']['val_loss_best_epoch']:.4f}")
    print("\nEl conjunto de test no se utilizó en estos experimentos.")


def command_real_test(args: argparse.Namespace) -> None:
    header("FASE 11 · Prueba con fotografías reales")
    from src.real_test import run_real_photo_test

    result = run_real_photo_test(
        manifest_path=Path(args.manifest) if args.manifest else None,
        model_name=args.model,
        min_area=args.min_area,
    )
    if result["estado"] == "sin_fotografias":
        print(result["mensaje"])
        print(f"\nPlantilla de manifiesto: {result['manifest']}")
        print(f"Instrucciones: {result['instrucciones']}")
    else:
        print(f"Fotografías procesadas: {result['n_fotos']}")
        if result.get("sequence_accuracy") is not None:
            print(f"Secuencias correctas: {result['sequence_accuracy'] * 100:.2f} %")
        print(f"Resultados: {result['json']}")
        print(f"Informe: {result['markdown']}")


def command_predict(args: argparse.Namespace) -> None:
    header("Predicción de una imagen")
    from src.prediction import predict_image

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    if not model_exists(args.model):
        raise FileNotFoundError(
            f"El modelo '{args.model}' no está entrenado. "
            f"Ejecuta: python main.py train --model {args.model}"
        )

    model = load_trained_model(args.model)
    prediction = predict_image(image_path, model, args.model, min_area=args.min_area)

    print(f"Modelo: {config.MODEL_DISPLAY_NAMES[args.model]}")
    print(f"Imagen: {image_path}")
    print(f"Dígitos detectados: {len(prediction.digits)}")
    print(f"Número reconocido: {prediction.number or '(ninguno)'}")
    print(f"Confianza media: {prediction.mean_confidence * 100:.2f} %")
    if prediction.fallback:
        print("Nota: no se segmentaron componentes; se procesó la imagen completa.")
    print("\nDetalle por dígito:")
    for item in prediction.predictions:
        print(
            f"  #{item.index + 1}: {item.digit} "
            f"({item.confidence * 100:.2f} %)"
        )

    import cv2

    output = config.RESULTS_DIR / f"prediccion_{image_path.stem}.png"
    cv2.imwrite(str(output), prediction.annotated_image)
    print(f"\nImagen anotada guardada en: {output}")


def command_info(args: argparse.Namespace) -> None:
    header("Información del entorno y del proyecto")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Ejecutable: {sys.executable}")
    print(f"Entorno virtual activo: {'sí' if sys.prefix != sys.base_prefix else 'no'}")
    print(f"Sistema: {platform.platform()}")

    print("\nBibliotecas:")
    packages = [
        "tensorflow",
        "keras",
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "pillow",
        "streamlit",
        "joblib",
    ]
    for package in packages:
        print(f"  {package:16s} {_package_version(package)}")

    if config.DATASET_INFO_FILE.exists():
        info = load_dataset_info()
        print("\nDataset:")
        print(f"  {info['nombre']}")
        print(f"  Entrenamiento: {info['conteos']['train']:,}")
        print(f"  Validación:    {info['conteos']['validation']:,}")
        print(f"  Prueba:        {info['conteos']['test']:,}")
        print(f"  Dimensiones:   {info['dimensiones_imagen'][0]}x{info['dimensiones_imagen'][1]}")
    else:
        print("\nDataset: aún no preparado (ejecuta `python main.py prepare`).")

    print("\nModelos:")
    rows = []
    for name in TRAIN_ORDER:
        rows.append(
            {
                "Clave": name,
                "Modelo": config.MODEL_DISPLAY_NAMES[name],
                "Entrenado": "sí" if model_exists(name) else "no",
                "Tamaño": _model_size(name),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    if config.METRICS_FILE.exists():
        print("\nÚltimas métricas de test:")
        print(get_metrics_table(load_metrics()).to_string(index=False))


def command_all(args: argparse.Namespace) -> None:
    command_prepare(args)
    if not hasattr(args, "model"):
        args.model = ["all"]
    if not hasattr(args, "skip_existing"):
        args.skip_existing = False
    command_train(args)
    evaluate_args = argparse.Namespace(models=None, no_cv=False)
    command_evaluate(evaluate_args)
    demo_args = argparse.Namespace(model=None)
    command_demo(demo_args)
    write_summary()


def _to_markdown(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a tabla Markdown sin depender de `tabulate`."""
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.values:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary() -> None:
    """Genera results/RESUMEN.md con los resultados reales de la última ejecución."""
    if not config.METRICS_FILE.exists():
        return

    metrics = load_metrics()
    table = get_metrics_table(metrics)
    info = load_dataset_info()

    lines = [
        "# Resumen de resultados",
        "",
        f"Generado: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> Todas las cifras provienen de ejecuciones reales sobre el conjunto de test "
        "oficial de MNIST (10 000 imágenes). No hay valores simulados.",
        "",
        "## Dataset",
        "",
        f"- Nombre: {info['nombre']}",
        f"- Entrenamiento: {info['conteos']['train']:,} imágenes",
        f"- Validación: {info['conteos']['validation']:,} imágenes",
        f"- Prueba: {info['conteos']['test']:,} imágenes",
        "- Dimensiones: 28x28, escala de grises",
        "",
        "## Métricas de test",
        "",
        _to_markdown(table),
        "",
    ]

    if config.KMEANS_METRICS_FILE.exists():
        from src.utils import load_json

        km = load_json(config.KMEANS_METRICS_FILE)
        lines += [
            "## K-Means (agrupamiento no supervisado)",
            "",
            f"- Clústeres: {km['n_clusters']}",
            f"- Adjusted Rand Index (ARI): {km['adjusted_rand_index']:.4f}",
            f"- Información mutua normalizada (NMI): {km['normalized_mutual_info']:.4f}",
            f"- Coeficiente de silueta: {km['silhouette']:.4f}",
            f"- Pureza: {km['purity'] * 100:.2f} %",
            f"- Exactitud tras mapeo clúster→dígito: {km['exactitud_tras_mapeo'] * 100:.2f} %",
            "",
            "K-Means no utiliza etiquetas durante el entrenamiento. La última cifra "
            "es solo una interpretación a posteriori.",
            "",
        ]

    if config.CV_RESULTS_FILE.exists():
        cv = pd.read_csv(config.CV_RESULTS_FILE)
        cv_view = cv[
            [
                "nombre",
                "n_muestras",
                "accuracy_mean",
                "accuracy_std",
                "f1_macro_mean",
                "f1_macro_std",
            ]
        ].copy()
        lines += ["## Validación cruzada (3 folds)", "", _to_markdown(cv_view), ""]

    if config.DEMO_RESULTS_FILE.exists():
        from src.utils import load_json

        demo = load_json(config.DEMO_RESULTS_FILE)
        lines += [
            "## Prueba del pipeline completo",
            "",
            f"- Modelo: {demo.get('model_display', '—')}",
            f"- Secuencias correctas: {demo['sequence_accuracy'] * 100:.1f} %",
            f"- Muestras: {len(demo['results'])}",
            "",
            demo.get("note", ""),
            "",
        ]

    config.SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen guardado en: {config.SUMMARY_FILE}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sistema Inteligente de Reconocimiento y Clasificación de Dígitos "
            "Manuscritos (MNIST)"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Descarga y divide MNIST")
    prepare.add_argument("--force", action="store_true", help="Regenera la división")
    prepare.set_defaults(func=command_prepare)

    train = subparsers.add_parser("train", help="Entrena uno o todos los modelos")
    train.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help="all o claves: logreg knn svm kmeans mlp cnn",
    )
    train.add_argument(
        "--skip-existing",
        action="store_true",
        help="No reentrena modelos que ya existen",
    )
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Evalúa y genera gráficos")
    evaluate.add_argument("--models", nargs="+", default=None)
    evaluate.add_argument("--no-cv", action="store_true", help="Omite validación cruzada")
    evaluate.set_defaults(func=command_evaluate)

    demo = subparsers.add_parser("demo", help="Prueba el pipeline con imágenes demo")
    demo.add_argument("--model", default=None)
    demo.set_defaults(func=command_demo)

    tune = subparsers.add_parser(
        "tune-cnn",
        help="Ejecuta experimentos reales de hiperparámetros de la CNN",
    )
    tune.add_argument("--experiments", nargs="+", default=None)
    tune.add_argument("--epochs", type=int, default=5)
    tune.add_argument("--subset", type=int, default=20_000)
    tune.add_argument("--quiet", action="store_true")
    tune.set_defaults(func=command_tune_cnn)

    real_test = subparsers.add_parser(
        "real-test",
        help="Prueba el sistema con fotografías reales descritas en un manifiesto",
    )
    real_test.add_argument("--manifest", default=None)
    real_test.add_argument("--model", default="cnn")
    real_test.add_argument("--min-area", type=int, default=config.DEFAULT_MIN_AREA)
    real_test.set_defaults(func=command_real_test)

    predict = subparsers.add_parser("predict", help="Predice una imagen propia")
    predict.add_argument("--image", required=True)
    predict.add_argument("--model", default="cnn")
    predict.add_argument("--min-area", type=int, default=config.DEFAULT_MIN_AREA)
    predict.set_defaults(func=command_predict)

    info = subparsers.add_parser("info", help="Muestra entorno, dataset y modelos")
    info.set_defaults(func=command_info)

    all_cmd = subparsers.add_parser("all", help="Prepara, entrena, evalúa y prueba todo")
    all_cmd.add_argument("--force-data", action="store_true")
    all_cmd.add_argument("--skip-existing", action="store_true")
    all_cmd.set_defaults(func=command_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "all":
            args.force = args.force_data
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
