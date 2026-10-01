"""二维桌面虚拟抓取：展示坐标、两连杆逆运动学和抓取流程。无硬件控制。"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from utils import PROJECT_ROOT


# 以下尺寸只是虚拟场景的设定，不是你的手机或未来机械臂的标定结果。
IMAGE_SIZE = (640, 480)
BASE_PIXEL = (90, 260)
PIXELS_PER_METER = 800
LINK_1_M = 0.22
LINK_2_M = 0.20
TARGET_PIXEL = (315, 175)
PLACE_PIXEL = (220, 380)
HOME_PIXEL = (155, 260)
FRAME_COUNT = 90
VIDEO_FPS = 15


def pixel_to_world(pixel):
    """把虚拟桌面像素换成以机械臂底座为原点的平面米坐标。"""
    x = (pixel[0] - BASE_PIXEL[0]) / PIXELS_PER_METER
    y = (BASE_PIXEL[1] - pixel[1]) / PIXELS_PER_METER
    return x, y


def world_to_pixel(x, y):
    """将虚拟米坐标变回画面像素坐标。"""
    return (round(BASE_PIXEL[0] + x * PIXELS_PER_METER),
            round(BASE_PIXEL[1] - y * PIXELS_PER_METER))


def inverse_kinematics(x, y):
    """计算虚拟平面二连杆的两个关节角，单位为弧度。"""
    distance = math.hypot(x, y)
    if distance > LINK_1_M + LINK_2_M or distance < abs(LINK_1_M - LINK_2_M):
        raise ValueError("目标超出虚拟机械臂的平面可达范围")

    # 根据余弦定理先求肘关节角，再求肩关节角。
    cos_elbow = (x * x + y * y - LINK_1_M**2 - LINK_2_M**2) / (2 * LINK_1_M * LINK_2_M)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
    shoulder = math.atan2(y, x) - math.atan2(
        LINK_2_M * math.sin(elbow), LINK_1_M + LINK_2_M * math.cos(elbow)
    )
    return shoulder, elbow


def forward_kinematics(shoulder, elbow):
    """从关节角算出肘部和夹爪在虚拟桌面上的坐标。"""
    joint = (LINK_1_M * math.cos(shoulder), LINK_1_M * math.sin(shoulder))
    hand = (joint[0] + LINK_2_M * math.cos(shoulder + elbow),
            joint[1] + LINK_2_M * math.sin(shoulder + elbow))
    return joint, hand


def interpolate(start, end, ratio):
    """在两组关节角之间做最简单的线性插值。"""
    return tuple(a + (b - a) * ratio for a, b in zip(start, end))


def frame_state(index, home_angles, target_angles, place_angles):
    """把 90 帧分为靠近、闭合、搬运、释放、退回五步。"""
    if index < 30:
        return interpolate(home_angles, target_angles, index / 29), False, "APPROACH"
    if index < 40:
        return target_angles, index >= 35, "CLOSE GRIPPER"
    if index < 70:
        return interpolate(target_angles, place_angles, (index - 40) / 29), True, "MOVE TO PLACE"
    if index < 80:
        return place_angles, False, "RELEASE"
    return interpolate(place_angles, home_angles, (index - 80) / 9), False, "RETURN HOME"


def draw_frame(index, home_angles, target_angles, place_angles):
    """绘制一帧；目标移动只是一段模拟动画，不代表真实抓取成功。"""
    canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), (38, 43, 49), dtype=np.uint8)
    cv2.rectangle(canvas, (15, 45), (625, 460), (70, 76, 80), 2)
    cv2.putText(canvas, "VIRTUAL TABLETOP - 2D ONLY", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
    cv2.putText(canvas, "No real camera calibration / no robot hardware", (20, 453),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    angles, grip_closed, stage = frame_state(index, home_angles, target_angles, place_angles)
    elbow_world, hand_world = forward_kinematics(*angles)
    elbow_pixel = world_to_pixel(*elbow_world)
    hand_pixel = world_to_pixel(*hand_world)

    # 只有闭合夹爪且进入搬运阶段后，虚拟物体才跟随夹爪。
    if index < 40:
        object_pixel = TARGET_PIXEL
    elif index < 70:
        object_pixel = hand_pixel
    else:
        object_pixel = PLACE_PIXEL

    cv2.circle(canvas, PLACE_PIXEL, 18, (70, 180, 70), 2)
    cv2.putText(canvas, "PLACE", (PLACE_PIXEL[0] + 22, PLACE_PIXEL[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 240, 150), 1)
    cv2.circle(canvas, object_pixel, 12, (0, 150, 255), -1)
    cv2.line(canvas, BASE_PIXEL, elbow_pixel, (220, 190, 90), 12)
    cv2.line(canvas, elbow_pixel, hand_pixel, (220, 190, 90), 9)
    cv2.circle(canvas, BASE_PIXEL, 16, (245, 245, 245), -1)
    cv2.circle(canvas, elbow_pixel, 10, (245, 245, 245), -1)
    cv2.circle(canvas, hand_pixel, 11 if grip_closed else 18,
               (50, 220, 70) if grip_closed else (100, 220, 220), 3)
    cv2.putText(canvas, f"Step: {stage}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)
    cv2.putText(canvas, f"Frame {index + 1}/{FRAME_COUNT}", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="无需机械臂的二维桌面虚拟抓取演示")
    parser.add_argument("--output", help="输出 MP4 路径；同时生成同名 JPG 预览图")
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (Path(args.output).expanduser().resolve() if args.output else
              PROJECT_ROOT / "results" / "videos" / f"virtual_tabletop_{timestamp}.mp4")
    preview = output.with_suffix(".jpg")
    if output.suffix.lower() != ".mp4" or output.exists() or preview.exists():
        print("错误：请使用新的 .mp4 输出文件名，不能覆盖已有视频或预览图。")
        return 1

    # 这里的坐标由预设的虚拟比例尺算出，并非真实相机标定。
    target_world = pixel_to_world(TARGET_PIXEL)
    place_world = pixel_to_world(PLACE_PIXEL)
    home_angles = inverse_kinematics(*pixel_to_world(HOME_PIXEL))
    target_angles = inverse_kinematics(*target_world)
    place_angles = inverse_kinematics(*place_world)
    print(f"虚拟目标：像素 {TARGET_PIXEL} -> 桌面坐标 {target_world} 米")
    print(f"虚拟放置点：像素 {PLACE_PIXEL} -> 桌面坐标 {place_world} 米")
    print(f"目标关节角：肩 {math.degrees(target_angles[0]):.1f}°, 肘 {math.degrees(target_angles[1]):.1f}°")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             VIDEO_FPS, IMAGE_SIZE)
    if not writer.isOpened():
        print(f"错误：无法创建视频：{output}")
        return 1
    try:
        for index in range(FRAME_COUNT):
            image = draw_frame(index, home_angles, target_angles, place_angles)
            writer.write(image)
            if index == 60 and not cv2.imwrite(str(preview), image):
                raise OSError(f"无法保存预览图：{preview}")
    finally:
        writer.release()
    print(f"虚拟演示完成：{output}")
    print(f"预览图：{preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
