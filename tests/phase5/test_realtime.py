"""Tests for real-time analytics endpoints."""
import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


MATCH_ID = "england_epl/2014-2015/match_001"
VIDEO_MATCH_ID = "england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal"

MOCK_VIDEO_INFO = {
    "fps": 25,
    "total_frames": 1000,
    "width": 1280,
    "height": 720,
    "duration_ms": 40000,
}

MOCK_FRAME_RESULT = {
    "frame_idx": 25,
    "timestamp_ms": 1000,
    "players": [
        {"track_id": 1, "bbox": [100, 200, 50, 80], "confidence": 0.95},
        {"track_id": 2, "bbox": [400, 300, 45, 75], "confidence": 0.88},
    ],
    "is_camera_cut": False,
}

MOCK_RUNNING_STATS = {
    "frames_processed": 50,
    "total_players_detected": 500,
    "avg_players_per_frame": 10.0,
    "avg_confidence": 0.87,
    "camera_cuts_detected": 2,
    "unique_track_ids": 22,
    "processing_fps": 15.3,
}


def _make_mock_session():
    """Create a mock session with sensible defaults for all methods."""
    session = MagicMock()
    session.get_video_info.return_value = MOCK_VIDEO_INFO
    session.process_at_time.return_value = MOCK_FRAME_RESULT
    session.process_at_frame.return_value = MOCK_FRAME_RESULT
    session.get_running_stats.return_value = MOCK_RUNNING_STATS
    session.processor = MagicMock()
    session.processor.get_video_info.return_value = MOCK_VIDEO_INFO
    return session


def _make_mock_manager(sessions=None, mock_session=None):
    """Create a mock SessionManager instance with the right return values."""
    mgr = MagicMock()
    mgr.list_sessions.return_value = sessions or []
    if mock_session:
        mgr.get_or_create.return_value = mock_session
        mgr.get_session.return_value = mock_session
    else:
        mgr.get_session.return_value = None
    return mgr


class TestRealtimeStatus:
    @patch("src.api.realtime_routes.SessionManager")
    def test_status_endpoint(self, mock_sm_cls, client):
        mock_sm_cls.return_value = _make_mock_manager()
        r = client.get("/realtime/status")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    @patch("src.api.realtime_routes.SessionManager")
    def test_status_empty_initially(self, mock_sm_cls, client):
        mock_sm_cls.return_value = _make_mock_manager()
        r = client.get("/realtime/status")
        assert r.status_code == 200
        data = r.json()
        assert data["sessions"] == []


class TestRealtimeStart:
    @patch("src.api.realtime_routes.SessionManager")
    @patch("src.api.realtime_routes.DATA_DIR")
    def test_start_session(self, mock_data_dir, mock_sm_cls, client):
        mock_session = _make_mock_session()
        mgr = _make_mock_manager(mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        # Make the filesystem checks pass
        mock_match_dir = MagicMock()
        mock_match_dir.is_dir.return_value = True
        mock_video_path = MagicMock()
        mock_video_path.is_file.return_value = True
        mock_match_dir.__truediv__ = MagicMock(return_value=mock_video_path)
        mock_data_dir.__truediv__ = MagicMock(return_value=mock_match_dir)

        r = client.post(f"/realtime/start/{VIDEO_MATCH_ID}")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "started"
        assert "video_info" in data
        assert data["video_info"]["fps"] == 25
        assert data["video_info"]["total_frames"] == 1000

    def test_start_nonexistent_match(self, client):
        r = client.post("/realtime/start/nonexistent")
        assert r.status_code == 404

    def test_start_match_no_video(self, client):
        r = client.post(f"/realtime/start/{MATCH_ID}")
        assert r.status_code == 404


class TestRealtimeFrame:
    @patch("src.api.realtime_routes.SessionManager")
    def test_get_frame(self, mock_sm_cls, client):
        mock_session = _make_mock_session()
        mgr = _make_mock_manager(sessions=[VIDEO_MATCH_ID], mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        r = client.get(f"/realtime/frame/{VIDEO_MATCH_ID}?timestamp_ms=1000")
        assert r.status_code == 200
        data = r.json()
        assert data["frame_idx"] == 25
        assert data["timestamp_ms"] == 1000
        assert len(data["players"]) == 2

    @patch("src.api.realtime_routes.SessionManager")
    def test_frame_requires_timestamp(self, mock_sm_cls, client):
        mock_session = _make_mock_session()
        mgr = _make_mock_manager(sessions=[VIDEO_MATCH_ID], mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        r = client.get(f"/realtime/frame/{VIDEO_MATCH_ID}")
        assert r.status_code == 400


class TestRealtimeStats:
    @patch("src.api.realtime_routes.SessionManager")
    def test_get_stats(self, mock_sm_cls, client):
        mock_session = _make_mock_session()
        mgr = _make_mock_manager(sessions=[VIDEO_MATCH_ID], mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        r = client.get(f"/realtime/stats/{VIDEO_MATCH_ID}")
        assert r.status_code == 200
        data = r.json()
        assert data["frames_processed"] == 50
        assert data["avg_players_per_frame"] == 10.0
        assert data["unique_track_ids"] == 22


class TestRealtimeStop:
    @patch("src.api.realtime_routes.SessionManager")
    def test_stop_session(self, mock_sm_cls, client):
        mgr = _make_mock_manager(sessions=[VIDEO_MATCH_ID])
        mock_sm_cls.return_value = mgr

        r = client.post(f"/realtime/stop/{VIDEO_MATCH_ID}")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "stopped"
        assert data["match_id"] == VIDEO_MATCH_ID


class TestWebSocket:
    @patch("src.api.websocket.SessionManager")
    @patch("src.api.websocket.DATA_DIR")
    def test_websocket_connect(self, mock_data_dir, mock_sm_cls, client):
        mock_session = _make_mock_session()

        mock_video_path = MagicMock()
        mock_video_path.exists.return_value = True
        mock_match_dir = MagicMock()
        mock_match_dir.__truediv__ = MagicMock(return_value=mock_video_path)
        mock_data_dir.__truediv__ = MagicMock(return_value=mock_match_dir)

        mgr = _make_mock_manager(mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        with client.websocket_connect(f"/ws/realtime/{VIDEO_MATCH_ID}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "connected"
            assert "video_info" in msg
            assert msg["video_info"]["fps"] == 25

    @patch("src.api.websocket.SessionManager")
    @patch("src.api.websocket.DATA_DIR")
    def test_websocket_seek(self, mock_data_dir, mock_sm_cls, client):
        mock_session = _make_mock_session()

        mock_video_path = MagicMock()
        mock_video_path.exists.return_value = True
        mock_match_dir = MagicMock()
        mock_match_dir.__truediv__ = MagicMock(return_value=mock_video_path)
        mock_data_dir.__truediv__ = MagicMock(return_value=mock_match_dir)

        mgr = _make_mock_manager(mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        with client.websocket_connect(f"/ws/realtime/{VIDEO_MATCH_ID}") as ws:
            ws.receive_json()  # consume connected
            ws.send_json({"type": "seek", "timestamp_ms": 1000})
            msg = ws.receive_json()
            assert msg["type"] == "frame_result"
            assert "data" in msg
            assert msg["data"]["frame_idx"] == 25

    @patch("src.api.websocket.SessionManager")
    @patch("src.api.websocket.DATA_DIR")
    def test_websocket_get_stats(self, mock_data_dir, mock_sm_cls, client):
        mock_session = _make_mock_session()

        mock_video_path = MagicMock()
        mock_video_path.exists.return_value = True
        mock_match_dir = MagicMock()
        mock_match_dir.__truediv__ = MagicMock(return_value=mock_video_path)
        mock_data_dir.__truediv__ = MagicMock(return_value=mock_match_dir)

        mgr = _make_mock_manager(mock_session=mock_session)
        mock_sm_cls.return_value = mgr

        with client.websocket_connect(f"/ws/realtime/{VIDEO_MATCH_ID}") as ws:
            ws.receive_json()  # consume connected
            ws.send_json({"type": "get_stats"})
            msg = ws.receive_json()
            assert msg["type"] == "stats"
            assert "data" in msg
            assert msg["data"]["frames_processed"] == 50
            assert msg["data"]["unique_track_ids"] == 22
