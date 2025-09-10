from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """统一的人体检测结果结构。

    - box_xyxy: 左上角与右下角坐标 (x1, y1, x2, y2)
    - confidence: 该目标为预测类别的置信度
    - class_id: 类别 ID（COCO 中 person 通常为 0）
    - class_name: 类别名称字符串（例如 "person"）
    """
    box_xyxy: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class YoloPersonDetector:
    """基于 Ultralytics YOLO 的人体检测器封装。

    设计要点：
    - 对 YOLO 推理进行轻量封装，只暴露与业务相关的参数与返回结构。
    - 仅保留对指定类别（默认 "person"）的目标，便于后续行为分类衔接。
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.35, iou_threshold: float = 0.5):
        # 加载 YOLO 模型（自动下载权重，或从本地路径加载）
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # 获取类别名称列表（COCO 数据集：person 类别通常为 0）
        self.class_names = self.model.model.names if hasattr(self.model.model, "names") else self.model.names

    def detect(self, frame_bgr: np.ndarray, target_class_names: Tuple[str, ...] = ("person",)) -> List[Detection]:
        """对单帧 BGR 图像进行人体检测，返回 Detection 列表。

        参数：
        - frame_bgr: OpenCV 读取的 BGR 帧
        - target_class_names: 需要保留的类别名称（默认只保留 "person"）
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes  # type: ignore[attr-defined]
        if boxes is None:
            return detections

        # 从 YOLO 结果中提取坐标、置信度与类别索引
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
            name = self.class_names.get(cid, str(cid)) if isinstance(self.class_names, dict) else self.class_names[cid]
            if name not in target_class_names:
                continue
            detections.append(
                Detection(
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    class_id=int(cid),
                    class_name=str(name),
                )
            )
        return detections


