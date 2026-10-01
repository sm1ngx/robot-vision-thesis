"""三个检测程序共同使用的简单工具函数。"""

from pathlib import Path
from typing import Any

import cv2


# 项目根目录。无论从哪个目录启动脚本，都能找到 models 和 results。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"


def calculate_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """根据检测框左上角和右下角坐标，计算中心点。"""
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y


def draw_detections(frame: Any, result: Any) -> tuple[Any, list[dict[str, Any]]]:
    """把一次 YOLO 检测结果画到画面上，并返回便于打印的数据。"""
    detections: list[dict[str, Any]] = []

    # result.boxes 中的每个 box 都代表一个检测到的目标。
    for box in result.boxes:
        # xyxy 表示检测框左上角 (x1, y1) 和右下角 (x2, y2)。
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        center_x, center_y = calculate_center(x1, y1, x2, y2)

        # 绿色矩形是 bounding box，红色圆点是目标中心点。
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        label = f"{class_name} {confidence:.2f} center=({center_x},{center_y})"
        text_y = max(y1 - 10, 20)
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "bounding_box": [x1, y1, x2, y2],
                "center": [center_x, center_y],
            }
        )

    return frame, detections


def print_detections(detections: list[dict[str, Any]], frame_number: int | None = None) -> None:
    """在终端中打印类别、置信度、检测框和中心坐标。"""
    prefix = f"第 {frame_number} 帧" if frame_number is not None else "图片"

    if not detections:
        print(f"{prefix}：没有检测到目标")
        return

    for index, detection in enumerate(detections, start=1):
        print(
            f"{prefix}目标 {index}: "
            f"类别={detection['class']}, "
            f"置信度={detection['confidence']:.3f}, "
            f"bounding_box={detection['bounding_box']}, "
            f"中心坐标={detection['center']}"
        )


def put_fps(frame: Any, fps: float) -> None:
    """在画面左上角显示当前处理速度。"""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def ensure_parent_directory(file_path: Path) -> None:
    """确保输出文件的父目录存在。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

