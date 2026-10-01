"""轨迹绘制的基本行为测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from robot_vision.trajectory import Trajectory


class TrajectoryTests(unittest.TestCase):
    def test_draws_line_for_connected_points(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        trajectory = Trajectory(max_length=3)
        trajectory.add((10, 10))
        trajectory.add((80, 80))
        trajectory.draw(image)

        self.assertGreater(int(image.sum()), 0)

    def test_keeps_only_configured_number_of_points(self) -> None:
        trajectory = Trajectory(max_length=2)
        trajectory.add((1, 1))
        trajectory.add((2, 2))
        trajectory.add((3, 3))

        self.assertEqual(list(trajectory.points), [(2, 2), (3, 3)])


if __name__ == "__main__":
    unittest.main()
