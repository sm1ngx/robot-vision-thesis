"""基于合成图像测试颜色目标检测，无需真实摄像头。"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from robot_vision.color_detector import ColorDetector


class ColorDetectorTests(unittest.TestCase):
    def test_detects_blue_circle_center(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 30, (255, 0, 0), -1)  # BGR 蓝色

        result = ColorDetector("blue", min_area=100).detect(frame)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.center[0], 160, delta=1)
        self.assertAlmostEqual(result.center[1], 120, delta=1)
        self.assertGreater(result.area, 2500)

    def test_ignores_small_blue_noise(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (80, 60), 3, (255, 0, 0), -1)

        self.assertIsNone(ColorDetector("blue", min_area=100).detect(frame))


if __name__ == "__main__":
    unittest.main()
