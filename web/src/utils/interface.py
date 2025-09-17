from __future__ import annotations

from typing import List, Tuple, Dict, Any

from ..pose.yolo11_pose_estimator import PoseDetection, KeyPoint, Yolo11PoseEstimator


def pack_pose_results(
    pose_detections: List[PoseDetection],
    behavior_results: List[Tuple[str, float]],
) -> List[Dict[str, Any]]:
    """将姿势估计和异常行为检测结果打包为标准 JSON 结构。
    
    应用层方案接口：基于YOLO11姿势估计的异常行为检测

    参数说明：
    - pose_detections: 姿势检测结果列表
    - behavior_results: 异常行为检测结果列表

    输出字段：
    - id: 以 1 开始的递增编号
    - box: [x1, y1, x2, y2] 人体检测框
    - person_confidence: 人体检测置信度（保留 1 位小数）
    - pose_confidence: 姿势估计置信度（保留 1 位小数）
    - keypoints: 17个关键点坐标和置信度
    - behavior: 异常行为检测结果
    - behavior_confidence: 异常行为检测置信度（保留 1 位小数）
    """
    
    results: List[Dict[str, Any]] = []
    
    for idx, (pose_det, behavior_result) in enumerate(zip(pose_detections, behavior_results)):
        # 构建关键点数据
        keypoints_data = []
        for i, kp in enumerate(pose_det.keypoints):
            keypoints_data.append({
                "id": i,
                "name": Yolo11PoseEstimator.KEYPOINT_NAMES[i] if i < len(Yolo11PoseEstimator.KEYPOINT_NAMES) else f"keypoint_{i}",
                "x": round(float(kp.x), 1),
                "y": round(float(kp.y), 1),
                "confidence": round(float(kp.confidence), 1)
            })
        
        # 构造单个目标的输出字典
        results.append({
            "id": idx + 1,
            "box": [
                int(pose_det.person_box[0]),
                int(pose_det.person_box[1]),
                int(pose_det.person_box[2]),
                int(pose_det.person_box[3]),
            ],
            "person_confidence": round(float(pose_det.person_confidence), 1),
            "pose_confidence": round(float(pose_det.pose_confidence), 1),
            "keypoints": keypoints_data,
            "behavior": str(behavior_result[0]),
            "behavior_confidence": round(float(behavior_result[1]), 1),
        })
    
    return results


# 备选方案接口函数（保留用于向后兼容）
def pack_results(
    detections: List[Any],  # 使用Any避免导入src1模块
    action_predictions: List[Tuple[str, float]] | None = None,
) -> List[Dict[str, Any]]:
    """将检测与分类结果打包为标准 JSON 结构（列表[dict]）。
    
    备选方案接口：保留用于向后兼容
    如需使用此函数，请从src1模块导入Detection类

    设计目的：
    - 将内部 `Detection` 与行为分类输出统一为简单、稳定、前后端通用的数据协议。
    - 该函数为唯一对外暴露的结果打包入口，便于版本演进时的向后兼容。

    参数说明：
    - detections: 检测模块输出的人体框结果列表。
    - action_predictions: 行为分类器输出，按与 detections 相同的顺序对齐，元素为 (label, confidence)。

    输出字段遵循 PRD 要求：
    - id: 以 1 开始的递增编号
    - box: [x1, y1, x2, y2]
    - confidence: 检测置信度，1 位小数
    - action: 行为类别（如 `NORMAL`/`P0`/`P1`/`P2`），无分类时为 `UNKNOWN`
    - action_confidence: 行为置信度，1 位小数，无分类时为 `0.0`
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