# 第三方来源与本项目工作划分

这份清单帮助在论文中准确区分“调用已有工具”“自己实现系统”和“将来需要完成的实验”。这里的“本项目代码”由 Codex 根据你的需求编写，属于当前项目中的整合与演示代码；它不意味着你本人已经逐行编写或训练过这些算法。论文中应如实标明你亲自完成的设计、修改、实验和分析。

| 项目中的内容 | 来源 | 实际作用 |
| --- | --- | --- |
| `models/yolo11n.pt` | [Ultralytics 官方 YOLO11 预训练权重](https://docs.ultralytics.com/models/yolo11/) | 识别已有类别；当前权重不是本项目训练的。 |
| `.venv` 中的 `ultralytics` | [Ultralytics 开源项目](https://github.com/ultralytics/ultralytics) | YOLO 推理和 ByteTrack 跟踪实现，项目脚本只是调用其 API。 |
| `.venv` 中的 `opencv-python` (`cv2`) | [OpenCV](https://opencv.org/license/) | 读取摄像头、图片、视频流，绘制框、文字并保存结果。 |
| `.venv` 中的 `torch` / `torchvision` | [PyTorch](https://github.com/pytorch/pytorch) | CPU 模型运算；本项目不修改这些库。 |
| `.venv` 中的 `lap` | [lap](https://github.com/gatagat/lap) | ByteTrack 的匹配计算依赖。 |
| `data/images/bus.jpg` | [Ultralytics 示例图片](https://ultralytics.com/images/bus.jpg) | 入门测试素材，不是你采集的数据。 |
| `data/videos/tracking_demo.mp4` | 本项目脚本由上述图片平移生成 | 功能验证用的合成视频，不是真实抓取数据。 |
| `configs/bytetrack.yaml` | 本项目设置的 ByteTrack 参数，参考 [Ultralytics 跟踪文档](https://docs.ultralytics.com/modes/track/) | 配置已有算法，不是从零实现 ByteTrack。 |
| `src/detect_*.py`、`src/track_*.py`、`src/utils.py`、`src/tracking*.py` | Codex 按你的毕设要求编写 | 输入、调用第三方模型、提取结果、绘图、输出与异常处理。 |
| `src/simulate_tabletop.py` | Codex 按此阶段问题编写 | 简化二维运动学与抓取动画，不是第三方物理引擎。 |
| `tests/`、`docs/`、`README.md` | Codex 按需求编写；测试素材注明上游来源 | 验证工程功能、记录学习过程和局限。 |

## 后续哪些工作适合由你自己完成

1. 明确目标物体和任务，例如“从桌面抓起杯子放入指定区域”，确定目标类别、夹爪大小、相机位置与评价指标。
2. 拍摄你自己的桌面场景；记录光照、遮挡、桌面背景，评估预训练模型是否能认出目标。论文中的核心实验数据应来自你的实际或明确标为仿真的场景。
3. 如果目标不在预训练类别中，收集并标注图片，划分训练/验证/测试集，再考虑微调轻量检测模型。拍图片不要求先有机械臂。
4. 设计相机标定、桌面坐标估计和坐标转换，并通过实测误差验证；这是从“看见”走到“抓取位置”的关键工作。
5. 有了具体机械臂后，针对其真实尺寸和运动限制建立模型、规划轨迹、评估抓取成功率。若采用学习式抓取策略，才需要相应动作、状态和成败数据；可先用仿真收集，最终仍需实物验证。
6. 在论文中分别描述采用的第三方算法、你实现的系统连接与改进、你的数据和实验结论。不要把 YOLO11、ByteTrack 或预训练权重写成“本人提出/训练”。

## 什么时候要训练自己的模型

并非必须。若预训练 YOLO11n 已能稳定识别目标，可以先用现成检测器加标定与预设抓取规则完成原型。若目标是特殊零件或在你的桌面场景中识别不稳，先采集**图片与标签**，对现有小模型做微调；不需要从零训练大模型。只有在任务目标要求学习抓取动作或姿态时，才进一步收集仿真或实物机械臂的操作数据。

## 引用与许可

学术论文需引用实际使用的算法、软件与图片来源。Ultralytics 官方将 YOLO11 代码和模型列为 [AGPL-3.0 / Enterprise 许可](https://docs.ultralytics.com/models/yolo11/)；[OpenCV 4.5 以上采用 Apache 2.0](https://opencv.org/license/)。本机安装包的元数据将 `lap` 标为 BSD-2-Clause，PyTorch wheel 列有多个许可组件。开源不等于没有使用条件。若今后公开发布代码、修改后的权重或商业化，应核对各项目当时的官方条款和学校要求；本清单是来源记录，不代替法律意见。
