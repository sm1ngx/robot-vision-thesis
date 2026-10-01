# 机器人视觉与具身智能本科毕设

本仓库将已有的机器人视觉工程、OpenCV 入门示例和毕设学习资料整理在一起。各模块保留独立入口与依赖，按学习阶段使用。

## 目录

| 目录 | 内容 | 使用说明 |
|---|---|---|
| `vision-workbench/` | OpenCV + YOLO11 物体检测、ByteTrack 跟踪、本地 Gradio 图片工作台及二维桌面仿真 | [视觉工作台 README](vision-workbench/README.md) |
| `opencv-basics/` | 摄像头颜色检测、物体中心坐标与运动轨迹 | [入门示例 README](opencv-basics/README.md) |
| `learning/` | 毕设路线图、学习进度、Python 练习与协作说明 | [学习资料 README](learning/README.md) |

当前工程已实现基础视觉检测、跟踪、图片网页界面与简化二维抓取动画。ROS 2、VLM、真实机械臂控制、三维定位及完整自主抓取系统属于后续计划，不是已经完成的功能。学习记录保留原更新时间，并不代表全部工程的最新状态。

## 运行视觉工作台

先克隆仓库并进入仓库根目录，然后在独立环境中运行：

```bash
cd vision-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

根据[视觉工作台说明](vision-workbench/README.md)准备 `models/yolo11n.pt` 后启动：

```bash
python app.py
```

浏览器访问 `http://127.0.0.1:7860`。模型权重、个人图片和视频、检测结果及本机虚拟环境未上传，需在本地准备。测试素材下载及其他检测、跟踪、仿真命令见模块 README。

## 运行 OpenCV 入门示例

另开终端，从仓库根目录运行：

```bash
cd opencv-basics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --camera 0
```

按 `q` 或 `Esc` 退出。无摄像头时可以运行合成图像测试：

```bash
python -m unittest discover -s tests -v
```

## 继续学习

从 [学习进度](learning/docs/学习进度.md)和 [路线图](learning/docs/路线图.md)开始；Python 练习在 `learning/python-learning/` 中。学习区的 [AGENTS.md](learning/AGENTS.md)说明教学偏好，仅作用于该学习目录。

## 来源与整理范围

此仓库是在本人需求下使用 Codex 辅助开发的学习与工程成果集合。OpenCV、YOLO、ByteTrack、PyTorch 和 Gradio 等第三方库与预训练模型的来源保持明确；不将这些算法或模型描述为本项目原创。

见 [整理来源](SOURCES.md)与[第三方来源和工作划分](vision-workbench/docs/third_party_and_authorship.md)。本次仅整理目录和运行说明，没有合并两个工程的内部实现，也没有删除原始项目文件夹。
