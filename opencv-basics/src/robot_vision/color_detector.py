"""使用 HSV 阈值检测单个颜色目标。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# HSV 对光照变化比 RGB 更稳健。红色跨越 Hue 的 0/180 边界，因此包含两段范围。
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "blue": [((100, 120, 70), (130, 255, 255))],
    "green": [((40, 70, 70), (85, 255, 255))],
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
}


@dataclass(frozen=True)
class Detection:
    """一次检测的几何结果。"""

    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area: float
    contour: np.ndarray


class ColorDetector:
    """寻找面积最大的指定颜色连通区域。"""

    def __init__(self, color: str = "blue", min_area: float = 400.0) -> None:
        if color not in COLOR_RANGES:
            raise ValueError(f"不支持的颜色：{color}")
        self.color = color
        self.min_area = min_area

    def create_mask(self, frame: np.ndarray) -> np.ndarray:
        """将 BGR 画面转换为经过形态学去噪的二值掩膜。"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in COLOR_RANGES[self.color]:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), dtype=np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def detect(self, frame: np.ndarray) -> Detection | None:
        """返回最大有效目标；没有符合条件的目标时返回 None。"""
        contours, _ = cv2.findContours(self.create_mask(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return None
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            return None
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        return Detection(center=center, bbox=cv2.boundingRect(contour), area=area, contour=contour)
