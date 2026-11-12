"""Pruebas del manifiesto de fotografías reales."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
import src.real_test as real_test


class TestRealPhotoManifest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=config.BASE_DIR)
        self.root = Path(self.temp_dir.name)
        self.patches = [
            patch.object(real_test, "REAL_PHOTOS_DIR", self.root),
            patch.object(real_test, "DEFAULT_MANIFEST", self.root / "manifest.csv"),
            patch.object(real_test, "ANNOTATED_DIR", self.root / "anotadas"),
            patch.object(real_test, "REAL_RESULTS_JSON", self.root / "real_results.json"),
            patch.object(real_test, "REAL_RESULTS_MD", self.root / "REAL_PHOTOS.md"),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_workspace_creates_template(self):
        path = real_test.ensure_real_photo_workspace()
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        self.assertEqual(list(df.columns), real_test.MANIFEST_COLUMNS)
        self.assertTrue((self.root / "LEEME.md").exists())

    def test_empty_manifest_returns_pending_status(self):
        real_test.ensure_real_photo_workspace()
        payload = real_test.run_real_photo_test()
        self.assertEqual(payload["estado"], "sin_fotografias")
        self.assertEqual(payload["n_fotos"], 0)
        self.assertIsNone(payload["sequence_accuracy"])
        self.assertTrue((self.root / "real_results.json").exists())
        self.assertTrue((self.root / "REAL_PHOTOS.md").exists())

    def test_synthetic_image_is_rejected(self):
        real_test.ensure_real_photo_workspace()
        df = pd.DataFrame(
            [
                {
                    "archivo": "sintetica.png",
                    "esperado": "2026",
                    "modelo": "cnn",
                    "min_area": 40,
                    "tipo": "sintetica",
                    "descripcion": "No debe contarse como foto real",
                }
            ]
        )
        df.to_csv(real_test.DEFAULT_MANIFEST, index=False)
        payload = real_test.run_real_photo_test()
        self.assertEqual(payload["estado"], "sin_fotografias")
        self.assertEqual(payload["n_fotos"], 0)
        self.assertEqual(len(payload["omitidos"]), 1)
        self.assertIn("foto_real", payload["omitidos"][0]["motivo"])

    def test_manifest_missing_columns_raises(self):
        path = self.root / "incompleto.csv"
        pd.DataFrame([{"archivo": "foto.jpg"}]).to_csv(path, index=False)
        with self.assertRaises(ValueError):
            real_test.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
