"""Pruebas del preprocesamiento y del dataset."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

import config
from src.dataset import load_dataset
from src.preprocessing import (
    binarize_digit,
    center_by_mass,
    ensure_grayscale,
    normalize_pixels,
    prepare_dataset_for_cnn,
    prepare_dataset_for_ml,
    prepare_digit_for_model,
    resize_and_center,
    resize_digit,
)


class TestPreprocessing(unittest.TestCase):
    def test_grayscale_from_bgr(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :, 1] = 255  # verde en BGR
        gray = ensure_grayscale(image)
        self.assertEqual(gray.shape, (10, 10))
        self.assertTrue(np.all(gray > 0))

    def test_normalize_uint8(self):
        image = np.array([[0, 128, 255]], dtype=np.uint8)
        normalized = normalize_pixels(image)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized.max()), 1.0, places=6)
        self.assertAlmostEqual(float(normalized.min()), 0.0, places=6)

    def test_no_double_normalization(self):
        """Un float con valores ~1 no debe dividirse otra vez entre 255."""
        image = np.array([[0.0, 1.0, 1.0000001]], dtype=np.float32)
        normalized = normalize_pixels(image)
        self.assertGreater(float(normalized.max()), 0.99)

    def test_resize(self):
        image = np.zeros((100, 60), dtype=np.uint8)
        resized = resize_digit(image, 28)
        self.assertEqual(resized.shape, (28, 28))

    def test_binarize_dark_digit_on_light_background(self):
        image = np.full((100, 100), 255, dtype=np.uint8)
        cv2.rectangle(image, (35, 20), (65, 80), 0, -1)
        binary = binarize_digit(image)
        self.assertGreater(np.mean(binary[20:80, 35:65]), 200)
        self.assertEqual(binary[0, 0], 0)

    def test_binarize_light_digit_on_dark_background(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (35, 20), (65, 80), 255, -1)
        binary = binarize_digit(image)
        self.assertGreater(np.mean(binary[20:80, 35:65]), 200)
        self.assertEqual(binary[0, 0], 0)

    def test_center_by_mass(self):
        image = np.zeros((28, 28), dtype=np.float32)
        image[0:5, 0:5] = 1.0
        centered = center_by_mass(image)
        total = centered.sum()
        ys, xs = np.indices(centered.shape)
        cy = (ys * centered).sum() / total
        cx = (xs * centered).sum() / total
        self.assertAlmostEqual(cy, 13.5, delta=1.5)
        self.assertAlmostEqual(cx, 13.5, delta=1.5)

    def test_prepare_digit_for_model(self):
        image = np.full((120, 90, 3), 255, dtype=np.uint8)
        cv2.putText(image, "7", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
        prepared = prepare_digit_for_model(image)
        self.assertEqual(prepared.shape, (28, 28))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertGreaterEqual(float(prepared.min()), 0.0)
        self.assertLessEqual(float(prepared.max()), 1.0)
        self.assertGreater(int(np.count_nonzero(prepared)), 30)

    def test_resize_and_center(self):
        image = np.zeros((50, 30), dtype=np.float32)
        image[10:40, 5:25] = 1.0
        result = resize_and_center(image)
        self.assertEqual(result.shape, (28, 28))
        self.assertGreater(float(result.max()), 0.9)

    def test_batch_preparation(self):
        X = np.random.randint(0, 256, size=(4, 28, 28), dtype=np.uint8)
        ml = prepare_dataset_for_ml(X)
        cnn = prepare_dataset_for_cnn(X)
        self.assertEqual(ml.shape, (4, 784))
        self.assertEqual(cnn.shape, (4, 28, 28, 1))
        self.assertLessEqual(float(ml.max()), 1.0)
        self.assertLessEqual(float(cnn.max()), 1.0)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not config.DATA_FILE.exists():
            raise unittest.SkipTest("Dataset no preparado; ejecuta `python main.py prepare`.")
        cls.splits = load_dataset()

    def test_shapes_and_counts(self):
        self.assertEqual(self.splits["X_train"].shape, (54_000, 28, 28))
        self.assertEqual(self.splits["X_val"].shape, (6_000, 28, 28))
        self.assertEqual(self.splits["X_test"].shape, (10_000, 28, 28))
        self.assertEqual(self.splits["X_train"].dtype, np.uint8)

    def test_labels(self):
        for key in ("y_train", "y_val", "y_test"):
            labels = self.splits[key]
            self.assertEqual(labels.min(), 0)
            self.assertEqual(labels.max(), 9)
            self.assertEqual(len(labels), len(self.splits[key.replace("y_", "X_")]))

    def test_no_split_overlap_by_construction(self):
        """Los tamaños suman 70 000 y el test conserva sus 10 000 imágenes."""
        total = (
            len(self.splits["X_train"])
            + len(self.splits["X_val"])
            + len(self.splits["X_test"])
        )
        self.assertEqual(total, 70_000)


if __name__ == "__main__":
    unittest.main()
