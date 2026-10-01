# 阶段 2 验证记录

验证日期：2026-09-24。所有推理使用本项目虚拟环境与本地 YOLO11n 模型。

## 环境

- Python 3.10.12
- torch 2.14.0+cpu，torchvision 0.29.0+cpu
- ultralytics 8.4.157，opencv-python 4.14.0.94
- 本阶段唯一新增安装包：lap 0.5.12，放在 `.venv` 内
- `torch.cuda.is_available()` 为 False
- `pip check`：No broken requirements found
- 原有检测入口和模型文件未改动

## 实际推理验证

首先运行阶段 1 的 5 帧 `bus_test.mp4`，成功输出跟踪视频和 JSONL。

随后用 `tests/make_tracking_demo.py` 将阶段 1 官方 bus.jpg 缩放、平移，生成 60 帧、10 FPS、480×640 的演示。第 1～5 帧是空白，第 31～34 帧模拟暂时消失。它不是实际物体运动录像，更不是 MOT 精度基准。

执行命令：

```bash
python src/track_video.py --source data/videos/tracking_demo.mp4 --output results/videos/tracking_demo_tracked.mp4 --no-display
python tests/verify_tracking_demo.py --video results/videos/tracking_demo_tracked.mp4
```

注意：结果已存在，复现时请给 `--output` 换一个新名字。

检查结果：

- 60 条 JSONL 记录，帧号连续、时间戳正确。
- 实际逐帧解码输出 MP4，成功解码全部 60 帧，尺寸和帧率一致。
- 同一帧中的非空跟踪 ID 无重复。
- 空白帧的 `objects` 列表为空。
- 公交车在 50 个有效跟踪帧中保持 ID 1。
- 消失前最后一个公交车 ID 与恢复后第一个 ID 均为 1。
- 中心位置确实变化，具有超过 10 个不同的像素位置。
- 可视检查第 26 帧，ID、类别、置信度、中心位置、中心点和 FPS 正常绘制。

输出：

- `results/videos/tracking_demo_tracked.mp4`
- `results/videos/tracking_demo_tracked.jsonl`
- `results/images/tracking_demo_preview.jpg`

## 边界、窗口与依赖检查

`python -m unittest discover -s tests -v` 共 8 项通过：原有中心点测试 2 项，新增跟踪边界测试 6 项（空目标、未分配 ID、历史长度和过期清理、丢失间隔记录、已有输出保护、非法参数释放资源）。

另行实际调用命令，确认不存在的输入、输入输出相同、无摄像头均退出码为 1，并给出中文错误，没有 Python traceback。

实际打开了 OpenCV 窗口，测试通过替换 `waitKey` 返回值注入 q，确认 q 分支结束循环、保存已处理的一帧、关闭窗口并释放视频源。该检查验证程序退出路径，不是人工按键体验测试。

初次验证时测试环境没有 `/dev/video0`。之后用户在真实电脑上成功保存过 676 帧、约 49 秒的可读摄像头跟踪视频；另一次显示窗口无响应后进程被终止，导致 MP4 未正常关闭。窗口长期运行稳定性仍待验证。

后来新增 `track_stream.py`，使用本机 HTTP/MJPEG 服务模拟手机推流，实测 4 帧读取、YOLO11n 跟踪、MP4 解码与 JSONL 均正常。RTSP 接口由 OpenCV 支持，但尚未用用户的手机地址实测。验证脚本：`python tests/verify_stream.py`。

## 结果解读

这次验证证明基础流程、输出格式和简单场景下的跨帧关联能够运行。没有计算 MOTA、IDF1 等指标，没有验证多人交叉、长时遮挡、快速相机运动下的精度。ID 不是永久身份；ByteTrack 可能换 ID。二维中心点也不等于现实空间三维坐标。

第一帧模型初始化可能较慢，画面 FPS 仅统计推理和绘图。摄像头以标称帧率写视频时播放时间可能与真实经过时间不同，真实经过时间写在 JSONL 中。

参数参考：[Ultralytics 官方跟踪文档](https://docs.ultralytics.com/modes/track/)。源码使用已经安装的固定版本接口，后续不要仅为追新直接升级依赖。
