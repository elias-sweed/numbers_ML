"""Pruebas del historial local de reconocimientos."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.history import append_history, load_history


class TestHistory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "historial.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(load_history(self.path), [])

    def test_append_and_load_preserve_order(self):
        append_history({"numero": "12", "modelo": "CNN"}, path=self.path)
        append_history({"numero": "34", "modelo": "KNN"}, path=self.path)

        history = load_history(self.path)
        self.assertEqual([item["numero"] for item in history], ["34", "12"])
        self.assertTrue(all("fecha" in item for item in history))

    def test_limit_keeps_most_recent_entries(self):
        for value in range(5):
            append_history({"numero": str(value)}, path=self.path, max_entries=3)

        history = load_history(self.path)
        self.assertEqual([item["numero"] for item in history], ["4", "3", "2"])

    def test_corrupt_file_returns_empty_list(self):
        self.path.write_text("{esto no es json", encoding="utf-8")
        self.assertEqual(load_history(self.path), [])

    def test_non_list_or_non_dict_entries_are_ignored(self):
        self.path.write_text(json.dumps({"numero": "1"}), encoding="utf-8")
        self.assertEqual(load_history(self.path), [])

        self.path.write_text(json.dumps([{"numero": "1"}, "texto", 3]), encoding="utf-8")
        self.assertEqual(load_history(self.path), [{"numero": "1"}])

    def test_atomic_write_leaves_no_temporary_file(self):
        append_history({"numero": "7"}, path=self.path)
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
