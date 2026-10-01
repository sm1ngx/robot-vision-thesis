"""使用 YOLO11n 对一张图片进行目标检测。"""

import argparse
import os
from pathlib import Path

import cv2

# 把 Ultralytics 的运行设置保存在项目中，不写入系统级目录。
PROJECT_ROOT_FOR_CONFIG = Path(__file__).resolve().parent.parent
ULTRALYTICS_CONFIG_PARENT = PROJECT_ROOT_FOR_CONFIG / "configs" / "ultralytics"
ULTRALYTICS_CONFIG_PARENT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_PARENT))

from ultralytics import YOLO

from utils import (
    DEFAULT_MODEL_PATH,
    PROJECT_ROOT,
    draw_detections,
    ensure_parent_directory,
    print_detections,
)


def parse_args() -> argparse.Namespace:
    """读取用户在终端中输入的参数。"""
    parser = argparse.ArgumentParser(description="使用 YOLO11n 检测一张图片")
    parser.add_argument("--source", required=True, help="输入图片路径")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "images" / "detected_image.jpg"),
        help="结果图片路径",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="YOLO 模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="最低置信度，默认 0.25")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    # 先检查输入，给初学者明确错误信息，而不是让库抛出很长的异常。
    if not source_path.is_file():
        print(f"错误：找不到输入图片：{source_path}")
        return 1

    image = cv2.imread(str(source_path))
    if image is None:
        print(f"错误：OpenCV 无法读取该图片：{source_path}")
        return 1

    # 加载轻量 YOLO11n 模型，并明确要求在 CPU 上推理。
    model = YOLO(args.model)
    results = model.predict(image, conf=args.conf, device="cpu", verbose=False)

    annotated_image, detections = draw_detections(image.copy(), results[0])
    print_detections(detections)

    ensure_parent_directory(output_path)
    if not cv2.imwrite(str(output_path), annotated_image):
        print(f"错误：结果图片保存失败：{output_path}")
        return 1

    print(f"检测完成，结果已保存到：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
