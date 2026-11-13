"""Pruebas del pipeline de predicción sobre imágenes nuevas."""

from __future__ import annotations

import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.prediction import (
    load_image,
    predict_image,
    predict_probabilities,
    predict_single_digit,
    softmax,
)


class _DummyModel:
    """Modelo mínimo con interfaz scikit-learn para probar el pipeline."""

    classes_ = np.arange(10)

    def predict_proba(self, X):
        # Probabilidad alta para la clase 1 y uniforme para el resto.
        probs = np.full((len(X), 10), 0.02, dtype=np.float32)
        probs[:, 1] = 0.82
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class _DummyKerasModel:
    """Modelo Keras mínimo para comprobar las formas de entrada."""

    def __init__(self):
        self.seen_shape = None

    def predict(self, X, verbose=0):
        self.seen_shape = X.shape
        probs = np.full((len(X), 10), 0.01, dtype=np.float32)
        probs[:, 7] = 0.91
        return probs / probs.sum(axis=1, keepdims=True)


def _canvas_with_text(text: str, width: int = 500) -> np.ndarray:
    canvas = np.full((170, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        text,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        (0, 0, 0),
        8,
        cv2.LINE_AA,
    )
    return canvas


class TestPrediction(unittest.TestCase):
    def test_softmax(self):
        scores = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        probs = softmax(scores)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)
        self.assertEqual(int(np.argmax(probs)), 2)

    def test_load_image_from_bytes(self):
        canvas = _canvas_with_text("1", width=200)
        success, buffer = cv2.imencode(".png", canvas)
        self.assertTrue(success)
        loaded = load_image(buffer.tobytes())
        self.assertEqual(loaded.ndim, 3)
        self.assertEqual(loaded.shape[2], 3)

    def test_transparent_png_is_composed_on_white(self):
        image = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((25, 10, 35, 50), fill=(0, 0, 0, 255))
        loaded = load_image(image)
        self.assertGreater(float(loaded[0, 0].mean()), 200)   # fondo blanco
        self.assertLess(float(loaded[30, 30].mean()), 60)    # trazo oscuro

    def test_predict_multi_digit(self):
        canvas = _canvas_with_text("1 2 3")
        model = _DummyModel()
        prediction = predict_image(canvas, model, "logreg")
        self.assertEqual(len(prediction.digits), 3)
        self.assertEqual(prediction.number, "111")
        self.assertEqual(len(prediction.confidences), 3)
        self.assertFalse(prediction.fallback)
        self.assertGreater(prediction.mean_confidence, 0.5)
        for item in prediction.predictions:
            self.assertEqual(len(item.probabilities), 10)
            self.assertEqual(item.prepared.shape, (28, 28))

    def test_predict_blank_falls_back(self):
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        prediction = predict_image(blank, _DummyModel(), "logreg")
        self.assertTrue(prediction.fallback)
        self.assertEqual(len(prediction.digits), 1)

    def test_predict_single_digit(self):
        canvas = _canvas_with_text("5", width=200)
        result = predict_single_digit(canvas, _DummyModel(), "logreg")
        self.assertEqual(result.digit, 1)  # el dummy siempre predice 1
        self.assertAlmostEqual(result.confidence, 0.82 / (0.82 + 9 * 0.02), places=5)

    def test_probabilities_shape(self):
        canvas = _canvas_with_text("7", width=200)
        probs = predict_probabilities(_DummyModel(), "logreg", [np.zeros((28, 28), np.float32)])
        self.assertEqual(probs.shape, (1, 10))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_keras_mlp_uses_flattened_input(self):
        """La MLP debe recibir (N, 784), no la imagen 28x28x1 de la CNN."""
        model = _DummyKerasModel()
        probs = predict_probabilities(model, "mlp", [np.zeros((28, 28), np.float32)])
        self.assertEqual(model.seen_shape, (1, 784))
        self.assertEqual(probs.shape, (1, 10))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_keras_cnn_uses_image_input(self):
        """La CNN debe recibir (N, 28, 28, 1)."""
        model = _DummyKerasModel()
        probs = predict_probabilities(model, "cnn", [np.zeros((28, 28), np.float32)])
        self.assertEqual(model.seen_shape, (1, 28, 28, 1))
        self.assertEqual(probs.shape, (1, 10))


if __name__ == "__main__":
    unittest.main()
