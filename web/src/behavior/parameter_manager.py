"""
异常行为检测参数管理模块
负责异常行为检测参数的初始化、管理和应用
"""

from typing import Optional
from src.pose.yolo11_pose_estimator import Yolo11PoseEstimator


class BehaviorParameterManager:
    """异常行为检测参数管理器"""
    
    def __init__(self, config, pose_estimator: Yolo11PoseEstimator):
        """
        初始化参数管理器
        
        Args:
            config: 配置对象
            pose_estimator: 姿势估计器对象
        """
        self.config = config
        self.pose_estimator = pose_estimator
        self._apply_initial_parameters()
    
    def _apply_initial_parameters(self):
        """应用初始参数设置"""
        # 设置基础检测参数
        self.pose_estimator.set_speed_threshold(self.config.pose.speed_threshold)
        self.pose_estimator.set_min_keypoint_confidence(self.config.pose.min_keypoint_confidence)
        
        # 设置高级检测参数
        self.pose_estimator.set_acceleration_threshold(self.config.pose.acceleration_threshold)
        self.pose_estimator.set_arm_raise_angle_threshold(self.config.pose.arm_raise_angle_deg)
        
        # 设置决策和冷却参数
        if hasattr(self.config.pose, 'decision_window'):
            self.pose_estimator.set_decision_window(self.config.pose.decision_window)
        if hasattr(self.config.pose, 'cooldown_after_alert'):
            self.pose_estimator.set_cooldown_after_alert(self.config.pose.cooldown_after_alert)
    
    def update_speed_threshold(self, value: float):
        """更新速度阈值"""
        self.config.pose.speed_threshold = value
        self.pose_estimator.set_speed_threshold(value)
    
    def update_acceleration_threshold(self, value: float):
        """更新加速度阈值"""
        self.config.pose.acceleration_threshold = value
        self.pose_estimator.set_acceleration_threshold(value)
    
    def update_arm_raise_angle_threshold(self, value: float):
        """更新抬臂角度阈值"""
        self.config.pose.arm_raise_angle_deg = value
        self.pose_estimator.set_arm_raise_angle_threshold(value)
    
    def update_decision_window(self, value: int):
        """更新决策窗口大小"""
        self.config.pose.decision_window = value
        self.pose_estimator.set_decision_window(value)
    
    def update_cooldown_after_alert(self, value: int):
        """更新冷却时间"""
        self.config.pose.cooldown_after_alert = value
        self.pose_estimator.set_cooldown_after_alert(value)
    
    def get_current_parameters(self) -> dict:
        """获取当前所有参数"""
        return {
            'speed_threshold': self.config.pose.speed_threshold,
            'acceleration_threshold': self.config.pose.acceleration_threshold,
            'arm_raise_angle_deg': self.config.pose.arm_raise_angle_deg,
            'decision_window': self.config.pose.decision_window,
            'cooldown_after_alert': self.config.pose.cooldown_after_alert,
            'min_keypoint_confidence': self.config.pose.min_keypoint_confidence,
        }
    
    def reset_to_defaults(self):
        """重置为默认参数"""
        # 这里可以定义默认参数值，或者从配置文件重新加载
        self._apply_initial_parameters()
    
    def validate_parameters(self) -> bool:
        """验证参数的有效性"""
        try:
            # 检查参数范围
            if self.config.pose.speed_threshold < 0:
                return False
            if self.config.pose.acceleration_threshold < 0:
                return False
            if self.config.pose.arm_raise_angle_deg < 0 or self.config.pose.arm_raise_angle_deg > 180:
                return False
            if self.config.pose.decision_window < 1:
                return False
            if self.config.pose.cooldown_after_alert < 0:
                return False
            if self.config.pose.min_keypoint_confidence < 0 or self.config.pose.min_keypoint_confidence > 1:
                return False
            
            return True
        except Exception:
            return False
