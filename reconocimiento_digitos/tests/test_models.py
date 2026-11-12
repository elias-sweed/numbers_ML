"""Pruebas de construcción y entrenamiento mínimo de los modelos."""

from __future__ import annotations

import unittest

import numpy as np

from src.models import (
    build_cnn,
    build_kmeans,
    build_knn,
    build_logistic_regression,
    build_mlp,
    build_svm,
    count_parameters,
)


class TestModelBuilders(unittest.TestCase):
    def test_classic_models(self):
        self.assertIsNotNone(build_logistic_regression())
        self.assertIsNotNone(build_knn())
        self.assertIsNotNone(build_svm())

    def test_cnn_architecture(self):
        model = build_cnn()
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertGreater(count_parameters(model), 100_000)
        layer_names = [layer.name for layer in model.layers]
        for expected in ("conv_1", "pool_1", "flatten", "densa", "salida"):
            self.assertIn(expected, layer_names)

    def test_mlp_architecture(self):
        model = build_mlp()
        self.assertEqual(model.input_shape, (None, 784))
        self.assertEqual(model.output_shape, (None, 10))

    def test_cnn_custom_hyperparameters(self):
        """La CNN debe aceptar filtros, dropout y learning rate configurables."""
        model = build_cnn(
            filters=(16, 32),
            dropout_conv=0.15,
            dropout_dense=0.35,
            learning_rate=5e-4,
        )
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))
        conv_1 = model.get_layer("conv_1")
        conv_3 = model.get_layer("conv_3")
        self.assertEqual(conv_1.filters, 16)
        self.assertEqual(conv_3.filters, 32)
        self.assertAlmostEqual(float(model.get_layer("dropout_conv_1").rate), 0.15)
        self.assertAlmostEqual(float(model.get_layer("dropout_densa").rate), 0.35)

    def test_cnn_rejects_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            build_cnn(filters=(0, 64))
        with self.assertRaises(ValueError):
            build_cnn(dropout_conv=1.0)
        with self.assertRaises(ValueError):
            build_cnn(learning_rate=-1e-3)

    def test_logistic_regression_learns_toy_data(self):
        rng = np.random.default_rng(0)
        X = rng.random((300, 784), dtype=np.float32)
        y = rng.integers(0, 10, size=300)
        model = build_logistic_regression()
        model.set_params(max_iter=30)
        model.fit(X, y)
        predictions = model.predict(X[:20])
        self.assertEqual(predictions.shape, (20,))

    def test_kmeans_fits(self):
        rng = np.random.default_rng(0)
        X = rng.random((200, 784), dtype=np.float32)
        pipeline = build_kmeans()
        pipeline.set_params(pca__n_components=10, kmeans__n_clusters=3, kmeans__n_init=2)
        pipeline.fit(X)
        labels = pipeline.predict(X[:20])
        self.assertEqual(labels.shape, (20,))
        self.assertTrue(set(labels).issubset({0, 1, 2}))


if __name__ == "__main__":
    unittest.main()
