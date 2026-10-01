"""摄像头的打开、读取与释放。"""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """用上下文管理器确保摄像头在退出时被释放。"""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"无法打开摄像头 {self.index}")
        return self

    def read(self) -> np.ndarray:
        """读取一帧 BGR 图像；读取失败时抛出异常。"""
        if self.capture is None:
            raise RuntimeError("摄像头尚未打开")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("无法读取摄像头画面")
        return frame

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.capture is not None:
            self.capture.release()
