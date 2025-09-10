from __future__ import annotations

from typing import List, Tuple, Dict, Any

from ..detector.yolo_detector import Detection


def pack_results(
    detections: List[Detection],
    action_predictions: List[Tuple[str, float]] | None = None,
) -> List[Dict[str, Any]]:
    """将检测与分类结果打包为标准 JSON 结构（列表[dict]）。

    设计目的：
    - 将内部 `Detection` 与行为分类输出统一为简单、稳定、前后端通用的数据协议。
    - 该函数为唯一对外暴露的结果打包入口，便于版本演进时的向后兼容。

    参数说明：
    - detections: 检测模块输出的人体框结果列表。
    - action_predictions: 行为分类器输出，按与 detections 相同的顺序对齐，元素为 (label, confidence)。

    输出字段遵循 PRD 要求：
    - id: 以 1 开始的递增编号
    - box: [x1, y1, x2, y2]
    - confidence: 人体检测置信度（保留 1 位小数）
    - action: 行为分类结果（若无分类结果则为 "UNKNOWN"）
    - action_confidence: 行为分类置信度（保留 1 位小数，若无则为 0.0）
    """

    results: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections):
        # 默认值：当没有分类结果或长度不匹配时，使用占位信息
        action_label = "UNKNOWN"
        action_conf = 0.0
        if action_predictions and idx < len(action_predictions):
            action_label, action_conf = action_predictions[idx]

        # 构造单个目标的输出字典
        results.append(
            {
                "id": idx + 1,
                "box": [
                    int(det.box_xyxy[0]),
                    int(det.box_xyxy[1]),
                    int(det.box_xyxy[2]),
                    int(det.box_xyxy[3]),
                ],
                "confidence": round(float(det.confidence), 1),  # 数值格式化到 1 位小数
                "action": str(action_label),
                "action_confidence": round(float(action_conf), 1),  # 数值格式化到 1 位小数
            }
        )

    return results




