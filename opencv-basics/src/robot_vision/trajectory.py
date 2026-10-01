"""保存并绘制目标在连续画面中的运动轨迹。"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class Trajectory:
    """维护有限长度的中心点序列，None 会断开轨迹线。"""

    def __init__(self, max_length: int = 80) -> None:
        self.points: deque[tuple[int, int] | None] = deque(maxlen=max_length)

    def add(self, point: tuple[int, int] | None) -> None:
        """增加一帧的目标中心；未检测到目标时传入 None。"""
        self.points.append(point)

    def draw(self, image: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        """原地绘制连续的轨迹线，跳过丢失目标造成的断点。"""
        previous: tuple[int, int] | None = None
        for point in self.points:
            if previous is not None and point is not None:
                cv2.line(image, previous, point, color, 2)
            previous = point
