from dataclasses import dataclass, field


@dataclass
class VideoConfig:
    camera_index: int = 0
    window_name: str = "CoffeeGuard - Pose Estimation"
    display_fps: bool = True
    target_fps: int = 30  # 期望摄像头FPS
    max_grab_per_loop: int = 2  # 每循环额外grab帧数以刷新缓冲


@dataclass
class PoseConfig:
    """YOLO11姿势估计配置"""
    model_name: str = "yolo11s-pose.pt"  # YOLO11姿势估计模型
    confidence_threshold: float = 0.5  # 检测置信度阈值
    image_size: int = 640  # 推理输入尺寸（像素）
    device: str = "cpu"  # "cpu" 或 "cuda:0"
    speed_threshold: float = 0.08  # 归一化速度阈值（相对人体框对角线/帧）
    min_keypoint_confidence: float = 0.3  # 关键点最小置信度
    acceleration_threshold: float = 0.15  # 归一化加速度阈值（相对对角线/帧^2）
    arm_raise_angle_deg: float = 60.0  # 度
    decision_window: int = 5  # 多数投票窗口大小
    cooldown_after_alert: int = 8  # 告警后冷却帧数
    alert_hold_frames: int = 30  # 画面报警横幅保留帧数（无新异常时倒计时）
    alert_banner_text: str = "检测到异常行为"  # 报警横幅文本（中文）
    save_alert_snapshot: bool = True  # 发生首次报警时保存截图
    alert_snapshot_subdir: str = "alerts"  # 截图子目录（位于 logs/ 下）
    use_chinese_font: bool = True  # 是否用中文字体渲染横幅
    chinese_font_path: str = r"C:\\Windows\\Fonts\\simhei.ttf"  # Windows常见字体


@dataclass
class AppConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)


CONFIG = AppConfig()