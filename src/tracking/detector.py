"""YOLOv8 player detection wrapper."""
import numpy as np


class PlayerDetector:
    """Detects players in video frames using YOLOv8.

    Filters for person class (COCO index 0) and optionally by confidence
    and bounding box size to exclude non-players (referees at edges, etc).
    """
    def __init__(self, model_name="yolov8m.pt", confidence=0.25, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.device = device

    def detect(self, frame):
        """Run detection on a single frame.

        Args:
            frame: (H, W, 3) uint8 numpy array (BGR or RGB)

        Returns dict with:
            boxes: (N, 4) xyxy float32 array
            confidences: (N,) float32 array
            class_ids: (N,) int array (all zeros = person)
        """
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
        """Run detection on multiple frames."""
        return [self.detect(f) for f in frames]
