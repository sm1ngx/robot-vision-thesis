# 机器人视觉入门项目

一个使用 Python 与 OpenCV 的最小化机器人视觉示例。程序从摄像头读取实时画面，检测指定颜色的物体，计算物体中心坐标，并绘制该中心的运动轨迹。

## 功能

- 实时摄像头画面显示
- 基于 HSV 的颜色分割（默认检测蓝色）
- 最大有效目标的轮廓、边界框与中心坐标
- 跨帧运动轨迹绘制
- 无需摄像头的单元测试

## 安装与运行

建议先创建虚拟环境，然后安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

启动摄像头（`0` 是默认摄像头编号）：

```bash
python3 app.py --camera 0
```

按 `q` 或 `Esc` 退出。可用 `--color red` 或 `--color green` 切换预设颜色。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

## 目录结构

```text
.
├── app.py                     # 程序入口与实时处理循环
├── requirements.txt           # 第三方依赖
├── src/robot_vision/
│   ├── camera.py              # 摄像头资源管理
│   ├── color_detector.py      # HSV 颜色目标检测
│   ├── trajectory.py          # 轨迹点保存与绘制
│   └── visualization.py       # 结果叠加显示
└── tests/
    ├── test_color_detector.py # 颜色检测单元测试
    └── test_trajectory.py     # 轨迹模块单元测试
```

## 调参提示

环境光变化较大时，在 `src/robot_vision/color_detector.py` 的 `COLOR_RANGES` 中调整 HSV 阈值；也可以降低或提高 `min_area` 来过滤过小的噪声区域。
