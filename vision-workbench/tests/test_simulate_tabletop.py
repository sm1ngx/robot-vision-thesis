"""检查虚拟机械臂几何计算：逆解能到目标，超出臂长会拒绝。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from simulate_tabletop import (  # noqa: E402
    PLACE_PIXEL, TARGET_PIXEL, forward_kinematics, inverse_kinematics, pixel_to_world
)


class TestVirtualArm(unittest.TestCase):
    def test_target_and_place_are_reachable(self):
        for pixel in (TARGET_PIXEL, PLACE_PIXEL):
            x, y = pixel_to_world(pixel)
            angles = inverse_kinematics(x, y)
            _, hand = forward_kinematics(*angles)
            self.assertAlmostEqual(hand[0], x, places=6)
            self.assertAlmostEqual(hand[1], y, places=6)

    def test_unreachable_target_is_rejected(self):
        with self.assertRaises(ValueError):
            inverse_kinematics(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
