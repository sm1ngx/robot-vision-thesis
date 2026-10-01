"""第二阶段入口：对本地视频做多目标跟踪。"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from tracking import run_tracking
from utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="YOLO11n + ByteTrack CPU 视频跟踪")
    parser.add_argument("--source", required=True, help="本地输入视频")
    parser.add_argument("--output", help="输出 .mp4 路径；同时保存同名 .jsonl")
    parser.add_argument("--conf", type=float, default=0.10, help="检测阈值，默认 0.10")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640；CPU 较慢可试 416")
    parser.add_argument("--no-display", action="store_true", help="关闭窗口，仍保存结果")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数；0 表示全部")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        print(f"错误：找不到视频：{source}")
        return 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (Path(args.output) if args.output else
              PROJECT_ROOT / "results" / "videos" / f"{source.stem}_tracked_{timestamp}.mp4")
    if output.expanduser().resolve() == source:
        print("错误：输出路径不能与输入视频相同。")
        return 1

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        print(f"错误：无法打开视频：{source}")
        return 1
    return run_tracking(capture, output, show=not args.no_display,
                        conf=args.conf, imgsz=args.imgsz, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
