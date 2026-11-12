"""Pruebas de la configuración de experimentos de hiperparámetros."""

from __future__ import annotations

import unittest

from src.tuning import EXPERIMENTS, _select_experiments, _training_subset


class TestTuning(unittest.TestCase):
    def test_experiments_cover_all_factors(self):
        factors = {experiment["factor"] for experiment in EXPERIMENTS}
        self.assertIn("configuración base", factors)
        self.assertIn("learning rate", factors)
        self.assertIn("cantidad de filtros", factors)
        self.assertIn("dropout", factors)
        self.assertIn("batch size", factors)
        self.assertIn("base", {experiment["id"] for experiment in EXPERIMENTS})

    def test_select_experiments(self):
        selected = _select_experiments(["base", "lr_5e-4"])
        self.assertEqual([item["id"] for item in selected], ["base", "lr_5e-4"])

    def test_unknown_experiment_raises(self):
        with self.assertRaises(ValueError):
            _select_experiments(["no_existe"])

    def test_training_subset_is_stratified(self):
        import numpy as np

        rng = np.random.default_rng(0)
        X = rng.random((1000, 28, 28), dtype=np.float32)
        y = np.repeat(np.arange(10), 100)
        X_sub, y_sub = _training_subset(X, y, 300, random_state=42)
        self.assertEqual(X_sub.shape, (300, 28, 28))
        self.assertEqual(y_sub.shape, (300,))
        self.assertEqual(set(np.unique(y_sub)), set(range(10)))


if __name__ == "__main__":
    unittest.main()
