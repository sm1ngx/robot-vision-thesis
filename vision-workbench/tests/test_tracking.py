"""跟踪边界测试：未分配 ID、空画面、轨迹清理、输入输出保护。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tracking_utils import draw_tracks, update_history
from tracking import run_tracking


class TestTracking(unittest.TestCase):
    def test_missing_id_does_not_invent_identity(self):
        box = SimpleNamespace(xyxy=np.array([[10, 20, 90, 120]]),
                              id=None, cls=[0], conf=[0.8])
        result = SimpleNamespace(boxes=[box], names={0: "person"})
        history = {}
        _, objects = draw_tracks(np.zeros((160, 240, 3), np.uint8), result, history, 1)
        self.assertIsNone(objects[0]["track_id"])
        self.assertEqual(objects[0]["center"], [50, 70])
        self.assertEqual(history, {})

    def test_empty_frame_is_valid(self):
        result = SimpleNamespace(boxes=[], names={})
        _, objects = draw_tracks(np.zeros((160, 240, 3), np.uint8), result, {}, 1)
        self.assertEqual(objects, [])

    def test_history_is_bounded_and_expired_ids_are_removed(self):
        history = {}
        for frame in range(1, 101):
            update_history(history, [{"track_id": 7, "center": [frame, 50]}], frame)
        self.assertEqual(len(history[7]), 30)
        update_history(history, [], 130)
        self.assertEqual(history, {})

    def test_history_preserves_frame_gaps(self):
        history = {}
        update_history(history, [{"track_id": 2, "center": [10, 20]}], 1)
        update_history(history, [], 2)
        update_history(history, [{"track_id": 2, "center": [30, 20]}], 3)
        self.assertEqual([point[0] for point in history[2]], [1, 3])

    def test_existing_output_is_preserved_and_capture_released(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.mp4"
            output.write_bytes(b"keep this original data")
            capture = Mock()
            self.assertEqual(run_tracking(capture, output, show=False), 1)
            self.assertEqual(output.read_bytes(), b"keep this original data")
            capture.release.assert_called_once()
            capture.read.assert_not_called()

    def test_invalid_confidence_releases_capture(self):
        capture = Mock()
        self.assertEqual(run_tracking(capture, Path("unused.mp4"), show=False, conf=-1), 1)
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
