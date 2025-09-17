from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
from ultralytics import YOLO


@dataclass
class KeyPoint:
    """人体关键点数据结构"""
    x: float
    y: float
    confidence: float


@dataclass
class PoseDetection:
    """姿势检测结果结构"""
    person_box: Tuple[int, int, int, int]  # 人体检测框 (x1, y1, x2, y2)
    keypoints: List[KeyPoint]  # 17个关键点
    person_confidence: float  # 人体检测置信度
    pose_confidence: float  # 姿势估计置信度


class Yolo11PoseEstimator:
    """基于YOLO11s-pose的姿势估计器
    
    功能：
    - 使用YOLO11s-pose预训练模型进行人体检测和姿势估计
    - 检测17个人体关键点
    - 支持火柴人可视化
    - 基于关键点运动参数进行异常行为检测
    """
    
    # COCO关键点索引映射
    KEYPOINT_NAMES = [
        "nose",           # 0
        "left_eye",       # 1
        "right_eye",      # 2
        "left_ear",       # 3
        "right_ear",      # 4
        "left_shoulder",  # 5
        "right_shoulder", # 6
        "left_elbow",     # 7
        "right_elbow",    # 8
        "left_wrist",     # 9
        "right_wrist",    # 10
        "left_hip",       # 11
        "right_hip",      # 12
        "left_knee",      # 13
        "right_knee",     # 14
        "left_ankle",     # 15
        "right_ankle",    # 16
    ]
    
    # 关键点连接关系（用于绘制火柴人）
    SKELETON_CONNECTIONS = [
        # 头部
        (0, 1), (0, 2), (1, 3), (2, 4),
        # 躯干
        (5, 6), (5, 11), (6, 12), (11, 12),
        # 左臂
        (5, 7), (7, 9),
        # 右臂
        (6, 8), (8, 10),
        # 左腿
        (11, 13), (13, 15),
        # 右腿
        (12, 14), (14, 16),
    ]
    
    def __init__(self, model_name: str = "yolo11s-pose.pt", confidence_threshold: float = 0.5, image_size: int = 640, device: str | None = None):
        """初始化姿势估计器
        
        Args:
            model_name: YOLO11姿势估计模型名称
            confidence_threshold: 检测置信度阈值
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.image_size = int(image_size)
        self.device = device
        
        # 历史数据：按目标索引对齐
        self.previous_poses: List[PoseDetection] = []
        # 每个目标的关键点历史速度，用于加速度估计
        self.previous_keypoint_speeds: Dict[int, List[float]] = {}
        self.max_history_frames = 10
        # 行为时序与冷却
        self.behavior_history: Dict[int, List[str]] = {}
        self.cooldown_frames: Dict[int, int] = {}
        
        # 异常行为检测参数
        self.speed_threshold = 50.0  # 像素/帧（或像素/Δt），超过此速度认为异常
        self.acceleration_threshold = 80.0  # 像素/帧^2，超过认为异常（可调）
        self.arm_raise_angle_deg = 60.0  # 手臂上举角度阈值（相对竖直）
        self.min_keypoint_confidence = 0.3  # 关键点最小置信度
        # 决策参数
        self.decision_window = 5  # 多数投票窗口
        self.cooldown_after_alert = 8  # 告警后冷却帧数，抑制抖动
        
    def detect_poses(self, frame_bgr: np.ndarray) -> List[PoseDetection]:
        """检测图像中的人体姿势
        
        Args:
            frame_bgr: BGR格式的输入图像
            
        Returns:
            姿势检测结果列表
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False
        )
        
        pose_detections = []
        if not results:
            return pose_detections
            
        result = results[0]
        
        # 检查是否有检测结果
        if result.boxes is None or result.keypoints is None:
            return pose_detections
            
        boxes = result.boxes
        keypoints = result.keypoints
        
        # 提取检测框信息
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        
        # 提取关键点信息
        kpts = keypoints.xy.cpu().numpy()  # [N, 17, 2]
        kpts_conf = keypoints.conf.cpu().numpy()  # [N, 17]
        
        for i, ((x1, y1, x2, y2), conf) in enumerate(zip(xyxy, confs)):
            # 构建关键点列表
            person_keypoints = []
            for j in range(17):
                if j < len(kpts[i]) and j < len(kpts_conf[i]):
                    person_keypoints.append(KeyPoint(
                        x=float(kpts[i][j][0]),
                        y=float(kpts[i][j][1]),
                        confidence=float(kpts_conf[i][j])
                    ))
                else:
                    person_keypoints.append(KeyPoint(x=0.0, y=0.0, confidence=0.0))
            
            # 计算姿势置信度（所有关键点的平均置信度）
            valid_keypoints = [kp for kp in person_keypoints if kp.confidence > self.min_keypoint_confidence]
            pose_conf = np.mean([kp.confidence for kp in valid_keypoints]) if valid_keypoints else 0.0
            
            pose_detections.append(PoseDetection(
                person_box=(int(x1), int(y1), int(x2), int(y2)),
                keypoints=person_keypoints,
                person_confidence=float(conf),
                pose_confidence=float(pose_conf)
            ))
            
        return pose_detections
    
    def detect_abnormal_behavior(self, current_poses: List[PoseDetection], dt: float = 1.0) -> List[Tuple[str, float]]:
        """基于关键点运动检测异常行为
        
        Args:
            current_poses: 当前帧的姿势检测结果
            
        Returns:
            异常行为检测结果列表 [(behavior_type, confidence), ...]
        """
        # 首帧处理
        if not self.previous_poses:
            self.previous_poses = current_poses
            self.previous_keypoint_speeds = {}
            return [("NORMAL", 0.5) for _ in current_poses]

        # 通过 IoU 对齐当前检测与上一帧检测顺序，提升稳定性
        matched_indices = self._match_poses_by_iou(self.previous_poses, current_poses)

        behavior_results: List[Tuple[str, float]] = []
        for cur_idx, prev_idx in matched_indices:
            current_pose = current_poses[cur_idx]
            if prev_idx is None:
                behavior_results.append(("NORMAL", 0.5))
                continue
            previous_pose = self.previous_poses[prev_idx]

            # 速度与加速度
            max_speed, max_acc = self._compute_speeds_and_acc(previous_pose, current_pose, dt, cur_idx)

            # 角度（如手臂上举）
            angles = self._compute_key_angles(current_pose)
            arm_raise_flag = (
                (angles.get("left_arm_raise", 0.0) > self.arm_raise_angle_deg) or
                (angles.get("right_arm_raise", 0.0) > self.arm_raise_angle_deg)
            )

            # 基础规则融合得到即时判定
            instant_label = "NORMAL"
            instant_conf = 0.5
            if max_speed > self.speed_threshold:
                instant_label = "ABNORMAL_FAST_MOVEMENT"
                instant_conf = min(0.95, max_speed / (self.speed_threshold * 2.0))
            elif max_acc > self.acceleration_threshold:
                instant_label = "ABNORMAL_SUDDEN_ACCEL"
                instant_conf = min(0.9, max_acc / (self.acceleration_threshold * 2.0))
            elif arm_raise_flag:
                instant_label = "SUSPICIOUS_ARM_RAISE"
                instant_conf = 0.6

            # 冷却抑制：若仍在冷却期，则降级为NORMAL（除非为更高等级P0类，这里统一抑制）
            if self.cooldown_frames.get(cur_idx, 0) > 0 and instant_label != "NORMAL":
                instant_label, instant_conf = "NORMAL", 0.5

            # 多数投票：累积最近N帧标签
            hist = self.behavior_history.get(cur_idx, [])
            hist.append(instant_label)
            if len(hist) > self.decision_window:
                hist = hist[-self.decision_window:]
            self.behavior_history[cur_idx] = hist

            final_label = self._majority_label(hist)
            final_conf = instant_conf if final_label != "NORMAL" else 0.5

            # 若产生非NORMAL告警，设置冷却
            if final_label != "NORMAL":
                self.cooldown_frames[cur_idx] = self.cooldown_after_alert
            else:
                # 冷却计数自然衰减
                if self.cooldown_frames.get(cur_idx, 0) > 0:
                    self.cooldown_frames[cur_idx] = max(0, self.cooldown_frames[cur_idx] - 1)

            behavior_results.append((final_label, final_conf))

        # 更新历史
        self.previous_poses = current_poses
        # 截断历史速度长度在内部方法中处理
        return behavior_results
    
    def draw_pose_skeleton(self, frame: np.ndarray, pose_detection: PoseDetection, 
                          behavior_result: Tuple[str, float]) -> np.ndarray:
        """在图像上绘制火柴人骨架
        
        Args:
            frame: 输入图像
            pose_detection: 姿势检测结果
            behavior_result: 异常行为检测结果
            
        Returns:
            绘制后的图像
        """
        keypoints = pose_detection.keypoints
        box = pose_detection.person_box
        
        # 绘制人体检测框
        x1, y1, x2, y2 = box
        color = (0, 0, 255) if "ABNORMAL" in behavior_result[0] else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制关键点
        for kp in keypoints:
            if kp.confidence > self.min_keypoint_confidence:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 3, (255, 0, 0), -1)
        
        # 绘制骨架连接线
        for connection in self.SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                keypoints[start_idx].confidence > self.min_keypoint_confidence and
                keypoints[end_idx].confidence > self.min_keypoint_confidence):
                
                start_point = (int(keypoints[start_idx].x), int(keypoints[start_idx].y))
                end_point = (int(keypoints[end_idx].x), int(keypoints[end_idx].y))
                cv2.line(frame, start_point, end_point, (255, 255, 0), 2)
        
        # 绘制标签（中文显示）
        label_zh = self.map_label_to_zh(behavior_result[0])
        label = f"{label_zh} {behavior_result[1]:.1f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def set_speed_threshold(self, threshold: float):
        """设置速度阈值"""
        self.speed_threshold = threshold
    
    def set_min_keypoint_confidence(self, confidence: float):
        """设置关键点最小置信度"""
        self.min_keypoint_confidence = confidence

    def set_acceleration_threshold(self, threshold: float):
        self.acceleration_threshold = threshold

    def set_arm_raise_angle_threshold(self, angle_deg: float):
        self.arm_raise_angle_deg = angle_deg

    def set_decision_window(self, window: int):
        self.decision_window = max(1, int(window))

    def set_cooldown_after_alert(self, frames: int):
        self.cooldown_after_alert = max(0, int(frames))

    # --------------------- 内部辅助方法 ---------------------
    @staticmethod
    def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter if (area_a + area_b - inter) > 0 else 1e-6
        return inter / union

    def _match_poses_by_iou(self, prev: List[PoseDetection], cur: List[PoseDetection]) -> List[Tuple[int, Optional[int]]]:
        if not prev or not cur:
            return [(i, None) for i in range(len(cur))]
        matched: List[Tuple[int, Optional[int]]] = []
        used_prev: set[int] = set()
        for i, c in enumerate(cur):
            best_j = None
            best_iou = 0.0
            for j, p in enumerate(prev):
                if j in used_prev:
                    continue
                iou = self._box_iou(p.person_box, c.person_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j is not None and best_iou > 0.1:
                used_prev.add(best_j)
                matched.append((i, best_j))
            else:
                matched.append((i, None))
        return matched

    def _compute_speeds_and_acc(self, prev_pose: PoseDetection, cur_pose: PoseDetection, dt: float, cur_idx: int) -> Tuple[float, float]:
        dt = max(dt, 1e-6)
        max_speed = 0.0
        speeds: List[float] = []
        # 使用人体框对角线归一化速度/加速度，降低因摄像头抖动与距离差异带来的影响
        x1, y1, x2, y2 = cur_pose.person_box
        diag = float(np.hypot(x2 - x1, y2 - y1))
        diag = diag if diag > 1e-6 else 1.0
        for current_kp, prev_kp in zip(cur_pose.keypoints, prev_pose.keypoints):
            if current_kp.confidence > self.min_keypoint_confidence and prev_kp.confidence > self.min_keypoint_confidence:
                dx = current_kp.x - prev_kp.x
                dy = current_kp.y - prev_kp.y
                speed = (float(np.hypot(dx, dy)) / diag) / dt
                speeds.append(speed)
                if speed > max_speed:
                    max_speed = speed
        # 记录平均速度用于加速度估计
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        history = self.previous_keypoint_speeds.get(cur_idx, [])
        history.append(mean_speed)
        if len(history) > self.max_history_frames:
            history = history[-self.max_history_frames:]
        self.previous_keypoint_speeds[cur_idx] = history
        max_acc = 0.0
        if len(history) >= 2:
            # 近似加速度：相邻均速差/ dt
            diffs = [abs(history[k] - history[k - 1]) / dt for k in range(1, len(history))]
            max_acc = max(diffs) if diffs else 0.0
        return max_speed, max_acc

    def _compute_key_angles(self, pose: PoseDetection) -> Dict[str, float]:
        k = pose.keypoints
        def angle_with_vertical(a_idx: int, b_idx: int) -> float:
            if a_idx >= len(k) or b_idx >= len(k):
                return 0.0
            a, b = k[a_idx], k[b_idx]
            if a.confidence < self.min_keypoint_confidence or b.confidence < self.min_keypoint_confidence:
                return 0.0
            # 向量 from shoulder to wrist (or elbow)
            vx = b.x - a.x
            vy = a.y - b.y  # 图像坐标 y 向下，取反使上为正
            v = np.array([vx, vy], dtype=float)
            if np.linalg.norm(v) < 1e-6:
                return 0.0
            # 与竖直单位向量 [0,1] 的夹角（度）
            cos_theta = np.clip(v[1] / np.linalg.norm(v), -1.0, 1.0)
            return float(np.degrees(np.arccos(cos_theta)))

        angles: Dict[str, float] = {}
        # 左臂：肩(5)->腕(9)，右臂：肩(6)->腕(10)
        angles["left_arm_raise"] = angle_with_vertical(5, 9)
        angles["right_arm_raise"] = angle_with_vertical(6, 10)
        return angles

    @staticmethod
    def _majority_label(labels: List[str]) -> str:
        if not labels:
            return "NORMAL"
        counts: Dict[str, int] = {}
        for lb in labels:
            counts[lb] = counts.get(lb, 0) + 1
        # 返回计数最高的标签；若并列，优先非NORMAL
        sorted_items = sorted(counts.items(), key=lambda x: (x[1], x[0] == "NORMAL"))
        return sorted_items[-1][0]

    @staticmethod
    def map_label_to_zh(label: str) -> str:
        mapping = {
            "NORMAL": "正常",
            "ABNORMAL_FAST_MOVEMENT": "异常：快速移动",
            "ABNORMAL_SUDDEN_ACCEL": "异常：突然加速",
            "SUSPICIOUS_ARM_RAISE": "可疑：抬臂",
        }
        return mapping.get(label, label)
