"""第二阶段入口：摄像头多目标跟踪，按 q 或 Ctrl+C 退出。"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2

from tracking import run_tracking
from utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="YOLO11n + ByteTrack CPU 摄像头跟踪")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--output", help="输出 .mp4 路径；同时保存同名 .jsonl")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-display", action="store_true", help="关闭窗口，用 Ctrl+C 结束")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数；0 表示不限")
    args = parser.parse_args()

    # Ubuntu 使用 V4L2；设备不存在时直接提示，避免底层后端反复报错。
    if args.camera < 0 or not Path(f"/dev/video{args.camera}").exists():
        print(f"错误：未发现摄像头 /dev/video{args.camera}，请连接摄像头或修改 --camera。")
        return 1
    capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        print("错误：无法打开摄像头，请检查权限、设备编号或是否被其他程序占用。")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (Path(args.output) if args.output else
              PROJECT_ROOT / "results" / "videos" / f"camera_tracked_{timestamp}.mp4")
    print("本次会保存摄像头视频和逐帧 JSONL 记录。")
    return run_tracking(capture, output, camera=True, show=not args.no_display,
                        conf=args.conf, imgsz=args.imgsz, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
