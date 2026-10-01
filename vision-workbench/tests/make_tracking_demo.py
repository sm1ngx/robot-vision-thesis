"""利用阶段 1 的 bus.jpg 生成可复现的移动测试视频，不下载新素材。"""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    image = cv2.imread(str(ROOT / "data" / "images" / "bus.jpg"))
    if image is None:
        raise SystemExit("请先准备 data/images/bus.jpg（README 有来源说明）。")
    output = ROOT / "data" / "videos" / "tracking_demo.mp4"
    if output.exists():
        print(f"测试视频已存在，保留原文件：{output}")
        return
    image = cv2.resize(image, (480, 640))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 640))
    if not writer.isOpened():
        raise SystemExit("无法创建测试视频。")
    try:
        for index in range(60):
            # 1~5 帧空画面；31~34 帧暂时消失，其他帧缓慢平移原图。
            if index < 5 or 30 <= index < 34:
                frame = np.zeros_like(image)
            else:
                dx = int(24 * np.sin(index / 15))
                dy = int(12 * np.cos(index / 15))
                transform = np.float32([[1, 0, dx], [0, 1, dy]])
                frame = cv2.warpAffine(image, transform, (480, 640))
            writer.write(frame)
    finally:
        writer.release()
    print(f"已生成 60 帧 / 10 FPS / 480×640 的合成移动视频：{output}")
    print("这是由静态图像平移生成的功能测试，不是真实运动数据集或跟踪精度评测。")


if __name__ == "__main__":
    main()
