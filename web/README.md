# 无人咖啡机器人安防监控系统（本地开发）

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

## 运行摄像头人体检测（最小可运行）
```powershell
python run_webcam.py
```

- ESC 退出
- 默认使用 `yolov8n.pt` 模型；首次运行会自动下载权重。
- 按 `P`：打印当前帧的 JSON 结果到控制台。

### 接口输出（JSON 格式）
按 `P` 时输出如下结构（每个人一个对象）：

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

字段说明：
- `id`: 连续编号，从 1 开始。
- `box`: 人体边界框，像素坐标，`[x1, y1, x2, y2]`。
- `confidence`: 人体检测置信度，保留 1 位小数。
- `action`: 行为分类结果（占位版本固定为 `NORMAL`）。
- `action_confidence`: 行为分类置信度，保留 1 位小数。

## 接口规范（检测+分类输出）

- 返回对象：列表，每个元素为一个目标字典
- 字段定义：
  - `id`: 从 1 开始的递增编号
  - `box`: `[x1, y1, x2, y2]` 像素坐标
  - `confidence`: 检测置信度，1 位小数
  - `action`: 行为类别（如 `NORMAL`/`P0`/`P1`/`P2`），无分类时为 `UNKNOWN`
  - `action_confidence`: 行为置信度，1 位小数，无分类时为 `0.0`

示例：
```json
[
  {
    "id": 1,
    "box": [100, 50, 220, 300],
    "confidence": 0.9,
    "action": "NORMAL",
    "action_confidence": 0.5
  }
]
```

## 运行与调试提示

- 启动实时摄像头：
  ```bash
  python web/run_webcam.py
  ```
- 按 `P`：在控制台打印当前帧的标准 JSON 结果
- 按 `ESC`：退出
- 置信度与 FPS 均保留 1 位小数，便于观察

## 目录结构
```
web/
  requirements.txt
  README.md
  run_webcam.py
  src/
    config.py
    utils/
      vis.py
    detector/
      yolo_detector.py
```


