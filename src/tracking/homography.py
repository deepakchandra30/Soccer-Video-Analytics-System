"""Pitch coordinate projection via homography estimation."""
import cv2
import numpy as np


# FIFA standard pitch dimensions (meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# common pitch keypoints in (x, y) meters, origin at center
PITCH_KEYPOINTS = {
    "center": (0.0, 0.0),
    "top_left": (-52.5, -34.0),
    "top_right": (52.5, -34.0),
    "bottom_left": (-52.5, 34.0),
    "bottom_right": (52.5, 34.0),
    "left_penalty_spot": (-41.5, 0.0),
    "right_penalty_spot": (41.5, 0.0),
    "left_goal_top": (-52.5, -3.66),
    "left_goal_bottom": (-52.5, 3.66),
    "right_goal_top": (52.5, -3.66),
    "right_goal_bottom": (52.5, 3.66),
    "center_top": (0.0, -34.0),
    "center_bottom": (0.0, 34.0),
    # penalty area corners
    "left_pa_top": (-36.0, -20.16),
    "left_pa_bottom": (-36.0, 20.16),
    "right_pa_top": (36.0, -20.16),
    "right_pa_bottom": (36.0, 20.16),
}


class PitchHomography:
    """Homography between image pixels and pitch coordinates."""
    def __init__(self):
        self.H = None
        self.H_inv = None
        self.inlier_mask = None

    def estimate(self, image_points, pitch_points, method=cv2.RANSAC,
                 reproj_threshold=5.0):
        """Estimate homography from image<->pitch point pairs, needs >= 4 points."""
        if len(image_points) < 4:
            return False

        src = np.array(pitch_points, dtype=np.float64)
        dst = np.array(image_points, dtype=np.float64)

        self.H_inv, self.inlier_mask = cv2.findHomography(
            src, dst, method, reproj_threshold,
        )
        if self.H_inv is None:
            return False

        self.H = np.linalg.inv(self.H_inv)
        return True

    def project_to_pitch(self, pixel_points):
        """Project pixel coordinates to pitch coordinates in meters."""
        if self.H is None:
            raise ValueError("Homography not estimated yet — call estimate() first")

        pts = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.H)
        return projected.reshape(-1, 2)

    def project_to_image(self, pitch_points):
        """Project pitch coordinates back to pixel coordinates."""
        if self.H_inv is None:
            raise ValueError("Homography not estimated yet")

        pts = np.array(pitch_points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, self.H_inv)
        return projected.reshape(-1, 2)

    def is_valid(self):
        """Check if a valid homography is loaded."""
        return self.H is not None

    @staticmethod
    def bbox_foot_position(boxes):
        """Bottom-center of xyxy bounding boxes as foot positions."""
        if len(boxes) == 0:
            return np.empty((0, 2), dtype=np.float32)
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2
        y_bottom = boxes[:, 3]
        return np.stack([x_center, y_bottom], axis=1).astype(np.float32)
