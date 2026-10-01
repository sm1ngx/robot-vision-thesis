"""不需要额外测试框架即可运行的基础单元测试。"""

import sys
import unittest
from pathlib import Path


# 将 src 加入导入路径，便于测试其中的函数。
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import calculate_center  # noqa: E402


class TestCalculateCenter(unittest.TestCase):
    def test_even_coordinates(self) -> None:
        self.assertEqual(calculate_center(10, 20, 30, 40), (20, 30))

    def test_odd_coordinates_use_integer_pixels(self) -> None:
        self.assertEqual(calculate_center(1, 3, 10, 12), (5, 7))


if __name__ == "__main__":
    unittest.main()

