from dataclasses import dataclass


@dataclass
class VideoConfig:
    camera_index: int = 0
    window_name: str = "CoffeeGuard - Webcam"
    display_fps: bool = True


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"  # ultralytics 模型名称或本地路径
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    target_class_names: tuple[str, ...] = ("person",)


@dataclass
class AppConfig:
    video: VideoConfig = VideoConfig()
    detection: DetectionConfig = DetectionConfig()


CONFIG = AppConfig()


