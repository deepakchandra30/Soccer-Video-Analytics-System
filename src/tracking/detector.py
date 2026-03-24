"""YOLOv8 player detection wrapper."""
import numpy as np


class PlayerDetector:
    """YOLOv8-based player detector, filters for person class."""
    def __init__(self, model_name="yolov8m.pt", confidence=0.25, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device

    def detect(self, frame):
        """Detect players in a single frame, returns boxes/confidences/class_ids."""
        results = self.model(
            frame, classes=[0], conf=self.confidence,
            device=self.device, verbose=False,
        )
        boxes_obj = results[0].boxes

        if len(boxes_obj) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "confidences": np.empty(0, dtype=np.float32),
                "class_ids": np.empty(0, dtype=np.int32),
            }

        return {
            "boxes": boxes_obj.xyxy.cpu().numpy().astype(np.float32),
            "confidences": boxes_obj.conf.cpu().numpy().astype(np.float32),
            "class_ids": boxes_obj.cls.cpu().numpy().astype(np.int32),
        }

    def detect_batch(self, frames):
        """Detect players in multiple frames."""
        return [self.detect(f) for f in frames]
