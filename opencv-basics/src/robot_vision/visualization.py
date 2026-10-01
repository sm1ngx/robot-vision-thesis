"""将检测结果叠加到实时画面。"""

from __future__ import annotations

import cv2
import numpy as np

from robot_vision.color_detector import Detection


def draw_detection(frame: np.ndarray, detection: Detection | None) -> np.ndarray:
    """复制画面并标注轮廓、框、中心坐标和检测状态。"""
    output = frame.copy()
    if detection is None:
        cv2.putText(output, "Target: not found", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output

    x, y, width, height = detection.bbox
    cx, cy = detection.center
    cv2.drawContours(output, [detection.contour], -1, (0, 255, 0), 2)
    cv2.rectangle(output, (x, y), (x + width, y + height), (255, 0, 0), 2)
    cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(output, f"Center: ({cx}, {cy})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return output
