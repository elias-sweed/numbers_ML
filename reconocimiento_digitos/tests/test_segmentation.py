"""Pruebas de la segmentación con OpenCV."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.segmentation import (
    annotate_segments,
    prepare_full_image,
    segment_digits,
    segmentation_preview,
)


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


class TestSegmentation(unittest.TestCase):
    def test_segment_five_digits(self):
        canvas = _canvas_with_text("1 2 3 4 5")
        segments = segment_digits(canvas)
        self.assertEqual(len(segments), 5)
        xs = [segment.bbox[0] for segment in segments]
        self.assertEqual(xs, sorted(xs), "Los dígitos deben ordenarse de izquierda a derecha.")
        for segment in segments:
            self.assertEqual(segment.prepared.shape, (28, 28))
            self.assertEqual(segment.prepared.dtype, np.float32)
            self.assertGreater(int(np.count_nonzero(segment.prepared)), 20)

    def test_single_digit(self):
        canvas = _canvas_with_text("8", width=200)
        segments = segment_digits(canvas)
        self.assertEqual(len(segments), 1)

    def test_blank_image_returns_empty(self):
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertEqual(segment_digits(blank), [])

    def test_annotate(self):
        canvas = _canvas_with_text("42", width=250)
        segments = segment_digits(canvas)
        annotated = annotate_segments(canvas, segments)
        self.assertEqual(annotated.shape, canvas.shape)
        self.assertEqual(annotated.ndim, 3)

    def test_preview_is_binary(self):
        canvas = _canvas_with_text("3", width=200)
        preview = segmentation_preview(canvas)
        self.assertEqual(set(np.unique(preview)).issubset({0, 255}), True)
        self.assertGreater(int(np.count_nonzero(preview)), 50)

    def test_full_image_fallback(self):
        canvas = _canvas_with_text("6", width=200)
        prepared = prepare_full_image(canvas)
        self.assertEqual(prepared.shape, (28, 28))
        self.assertGreater(int(np.count_nonzero(prepared)), 30)


if __name__ == "__main__":
    unittest.main()
