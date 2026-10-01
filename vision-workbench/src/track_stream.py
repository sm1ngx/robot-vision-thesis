"""用手机提供的 HTTP/MJPEG 或 RTSP 视频流做目标跟踪。"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import cv2

from tracking import run_tracking
from utils import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="YOLO11n + ByteTrack 手机视频流跟踪")
    parser.add_argument("--url", help="手机提供的视频流地址；也可设 ROBOT_VISION_STREAM_URL")
    parser.add_argument("--output", help="输出 .mp4 路径；同时保存同名 .jsonl")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=416,
                        help="CPU 推理尺寸，默认 416；更清晰但更慢可试 640")
    parser.add_argument("--no-display", action="store_true", help="不弹窗口，用 Ctrl+C 结束")
    parser.add_argument("--max-frames", type=int, default=0, help="处理帧数上限，默认 0 为不限")
    args = parser.parse_args()

    # 手机软件给出的应是视频流地址，而不是网页中的播放器页面。
    url = args.url or os.getenv("ROBOT_VISION_STREAM_URL", "")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https", "rtsp", "rtsps"} or not parsed.hostname:
        print("错误：请提供完整的 HTTP(S) MJPEG 或 RTSP 视频流地址。")
        print("示例：python src/track_stream.py --url 'http://手机IP:端口/video'")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (Path(args.output) if args.output else
              PROJECT_ROOT / "results" / "videos" / f"phone_tracked_{timestamp}.mp4")

    # FFmpeg 后端支持 IP 视频流。超时可避免地址错误时无限等待。
    # 不打印 URL，防止带用户名和密码的地址出现在终端日志中。
    options = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
               cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, options)
    if not capture.isOpened():
        capture.release()
        print("错误：无法打开手机视频流。请核对流地址、手机和电脑网络，以及手机端是否正在推流。")
        return 1

    print("手机视频流已连接，准备开始 CPU 跟踪。")
    return run_tracking(capture, output, camera=True, show=not args.no_display,
                        conf=args.conf, imgsz=args.imgsz, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
