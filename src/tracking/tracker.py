"""ByteTrack multi-object tracker with camera cut detection."""
import cv2
import numpy as np
import supervision as sv


class CameraCutDetector:
    """Detects camera cuts by comparing frame histograms."""

    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.prev_hist = None

    def is_cut(self, frame):
        """Returns True if this frame is a camera cut."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)

        if self.prev_hist is None:
            self.prev_hist = hist
            return False

        score = cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        self.prev_hist = hist
        return score > self.threshold

    def reset(self):
        self.prev_hist = None


class PlayerTracker:
    """ByteTrack tracker that resets on camera cuts."""
    def __init__(self, track_threshold=0.25, lost_buffer=30,
                 match_threshold=0.8, frame_rate=25, cut_threshold=0.4):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_threshold,
            lost_track_buffer=lost_buffer,
            minimum_matching_threshold=match_threshold,
            frame_rate=frame_rate,
        )
        self.cut_detector = CameraCutDetector(threshold=cut_threshold)
        self.frame_rate = frame_rate

    def update(self, frame, detections_dict):
        """Process one frame, returns tracked boxes with IDs and cut flag."""
        is_cut = self.cut_detector.is_cut(frame)
        if is_cut:
            self.tracker.reset()

        boxes = detections_dict["boxes"]
        confs = detections_dict["confidences"]
        class_ids = detections_dict["class_ids"]

        if len(boxes) == 0:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "confidences": np.empty(0, dtype=np.float32),
                "tracker_ids": np.empty(0, dtype=np.int32),
                "is_camera_cut": is_cut,
            }

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confs,
            class_id=class_ids,
        )

        tracked = self.tracker.update_with_detections(detections)

        return {
            "boxes": tracked.xyxy.astype(np.float32),
            "confidences": tracked.confidence.astype(np.float32),
            "tracker_ids": tracked.tracker_id.astype(np.int32)
                           if tracked.tracker_id is not None
                           else np.empty(0, dtype=np.int32),
            "is_camera_cut": is_cut,
        }

    def reset(self):
        """Reset tracker and cut detector state."""
        self.tracker.reset()
        self.cut_detector.reset()
