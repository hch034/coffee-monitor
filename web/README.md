# 无人咖啡机器人安防监控系统（本地开发）

## 项目概述

本项目采用**基于YOLO11姿势估计的应用层开发方案**，通过分析人体关键点运动参数来检测异常行为。

### 技术方案
- **应用层方案（当前）**：使用YOLO11s-pose预训练模型进行人体检测和17个关键点估计，基于关键点运动速度等参数检测异常行为
- **备选方案**：自训练模型开发（保留在src1目录中）

## 环境准备（Windows）
1. 安装 Python 3.10（或3.9/3.11，建议64位）
2. 创建虚拟环境并安装依赖：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

> 如有NVIDIA显卡且需要GPU，请根据 `https://pytorch.org/get-started/locally/` 安装匹配CUDA版本的PyTorch，然后再安装其余依赖。

## 运行摄像头姿势估计与异常行为检测

### 应用层方案（推荐）
```powershell
python run_webcam.py
```

- ESC 退出
- 默认使用 `yolo11s-pose.pt` 模型；首次运行会自动下载权重。
- 按 `P`：打印当前帧的 JSON 结果到控制台。
- 显示火柴人骨架和异常行为检测结果

### 备选方案（传统检测+分类）
```powershell
python run_webcam_legacy.py
```

- 使用传统的YOLO8检测+行为分类两阶段方案
- 使用 `yolov8n.pt` 模型
- 适用于需要自训练模型的场景

## 接口输出（JSON 格式）

### 应用层方案输出
按 `P` 时输出如下结构（每个人一个对象）：

```json
[
  {
    "id": 1,
    "box": [x1, y1, x2, y2],
    "person_confidence": 0.9,
    "pose_confidence": 0.8,
    "keypoints": [
      {
        "id": 0,
        "name": "nose",
        "x": 100.0,
        "y": 50.0,
        "confidence": 0.9
      },
      ...
    ],
    "behavior": "NORMAL",
    "behavior_confidence": 0.5
  }
]
```

### 备选方案输出
```json
[
  {
    "id": 1,
    "box": [x1, y1, x2, y2],
    "confidence": 0.9,
    "action": "NORMAL",      
    "action_confidence": 0.5  
  }
]
```

## 字段说明

### 应用层方案字段
- `id`: 连续编号，从 1 开始。
- `box`: 人体边界框，像素坐标，`[x1, y1, x2, y2]`。
- `person_confidence`: 人体检测置信度，保留 1 位小数。
- `pose_confidence`: 姿势估计置信度，保留 1 位小数。
- `keypoints`: 17个人体关键点列表，包含坐标和置信度。
- `behavior`: 异常行为检测结果（如 `NORMAL`/`ABNORMAL_FAST_MOVEMENT`）。
- `behavior_confidence`: 异常行为检测置信度，保留 1 位小数。

### 备选方案字段
- `id`: 连续编号，从 1 开始。
- `box`: 人体边界框，像素坐标，`[x1, y1, x2, y2]`。
- `confidence`: 人体检测置信度，保留 1 位小数。
- `action`: 行为分类结果（占位版本固定为 `NORMAL`）。
- `action_confidence`: 行为分类置信度，保留 1 位小数。

## 运行与调试提示

- 启动应用层方案：
  ```bash
  python run_webcam.py
  ```
- 启动备选方案：
  ```bash
  python run_webcam_legacy.py
  ```
- 按 `P`：在控制台打印当前帧的标准 JSON 结果
- 按 `ESC`：退出
- 应用层方案显示火柴人骨架可视化
- 异常行为检测基于关键点运动速度

## 配置参数

在 `src/config.py` 中可以调整以下参数：

```python
@dataclass
class PoseConfig:
    model_name: str = "yolo11s-pose.pt"  # YOLO11姿势估计模型
    confidence_threshold: float = 0.5  # 检测置信度阈值
    speed_threshold: float = 50.0  # 异常行为速度阈值（像素/帧）
    min_keypoint_confidence: float = 0.3  # 关键点最小置信度
```

## 目录结构
```
web/
  requirements.txt
  README.md
  run_webcam.py              # 应用层方案主程序
  run_webcam_legacy.py       # 备选方案主程序
  src/                       # 应用层方案代码
    config.py
    pose/
      __init__.py
      yolo11_pose_estimator.py  # YOLO11姿势估计模块
    utils/
      vis.py
      interface.py  # 接口函数
  src1/                      # 备选方案代码
    __init__.py
    detector/
      __init__.py
      yolo_detector.py       # 传统YOLO检测器
    classifier/
      __init__.py
      base.py                # 分类器基类
      dummy_classifier.py    # 占位分类器
```

## 方案说明

### 应用层方案（src/目录）
- 基于YOLO11s-pose预训练模型
- 单阶段姿势估计+异常行为检测
- 火柴人可视化
- 基于关键点运动参数的异常行为检测
- 开发速度快，无需训练自定义模型

### 备选方案（src1/目录）
- 基于YOLO8检测+行为分类的两阶段方案
- 支持自训练模型开发
- 传统检测框可视化
- 适用于需要自定义训练的特定场景
- 保留完整的训练和优化能力