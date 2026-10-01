"""机器人视觉示例的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# 允许直接从项目根目录运行，无需安装为 Python 包。
sys.path.insert(0, str(Path(__file__).parent / "src"))

from robot_vision.camera import Camera
from robot_vision.color_detector import ColorDetector
from robot_vision.trajectory import Trajectory
from robot_vision.visualization import draw_detection


def parse_args() -> argparse.Namespace:
    """读取摄像头编号和目标颜色。"""
    parser = argparse.ArgumentParser(description="实时颜色目标跟踪")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument(
        "--color", choices=("blue", "green", "red"), default="blue", help="检测颜色"
    )
    return parser.parse_args()


def main() -> None:
    """逐帧检测目标，并在窗口中显示结果。"""
    args = parse_args()
    detector = ColorDetector(args.color)
    trajectory = Trajectory(max_length=80)

    try:
        with Camera(args.camera) as camera:
            print("摄像头已启动，按 q 或 Esc 退出。")
            while True:
                frame = camera.read()
                detection = detector.detect(frame)
                trajectory.add(detection.center if detection else None)

                display = draw_detection(frame, detection)
                trajectory.draw(display)
                cv2.imshow("Robot Vision - Color Tracking", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except RuntimeError as error:
        print(f"错误：{error}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
