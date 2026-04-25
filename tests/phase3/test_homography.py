"""Tests for pitch homography module."""
import numpy as np
import pytest

from src.tracking.homography import PitchHomography, PITCH_LENGTH, PITCH_WIDTH


class TestPitchHomography:
    def test_estimate_succeeds(self, pitch_image_correspondences):
        image_pts, pitch_pts = pitch_image_correspondences
        h = PitchHomography()
        assert h.estimate(image_pts, pitch_pts)
        assert h.is_valid()

    def test_estimate_fails_with_few_points(self):
        h = PitchHomography()
        assert not h.estimate([[0, 0]], [[0, 0]])

    def test_roundtrip_projection(self, pitch_image_correspondences):
        image_pts, pitch_pts = pitch_image_correspondences
        h = PitchHomography()
        h.estimate(image_pts, pitch_pts)
        # project image points to pitch and back
        projected_pitch = h.project_to_pitch(image_pts)
        reprojected_image = h.project_to_image(projected_pitch)
        np.testing.assert_allclose(reprojected_image, image_pts, atol=2.0)

    def test_project_without_estimate_raises(self):
        h = PitchHomography()
        with pytest.raises(ValueError):
            h.project_to_pitch([[100, 200]])

    def test_pitch_dimensions(self):
        assert PITCH_LENGTH == 105.0
        assert PITCH_WIDTH == 68.0


class TestBboxFootPosition:
    def test_foot_position(self):
        boxes = np.array([[100, 200, 150, 400]], dtype=np.float32)
        feet = PitchHomography.bbox_foot_position(boxes)
        assert feet.shape == (1, 2)
        assert feet[0, 0] == pytest.approx(125.0)  # x center
        assert feet[0, 1] == pytest.approx(400.0)  # y bottom

    def test_empty_boxes(self):
        feet = PitchHomography.bbox_foot_position(np.empty((0, 4)))
        assert feet.shape == (0, 2)
