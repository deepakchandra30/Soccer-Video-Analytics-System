"""Tests for FastAPI endpoints."""
import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


MATCH_ID = "england_epl/2014-2015/match_001"


class TestHealthCheck:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestMatches:
    def test_list_matches(self, client):
        r = client.get("/matches")
        assert r.status_code == 200
        matches = r.json()
        assert len(matches) >= 1
        assert "id" in matches[0]
        assert "name" in matches[0]

    def test_match_not_found(self, client):
        r = client.get("/matches/nonexistent/events")
        assert r.status_code == 404


class TestEvents:
    def test_get_events(self, client):
        r = client.get(f"/matches/{MATCH_ID}/events")
        assert r.status_code == 200
        events = r.json()
        assert len(events) > 0
        assert "label" in events[0]
        assert "half" in events[0]
        assert "position" in events[0]
        assert "confidence" in events[0]


class TestTracks:
    def test_get_tracks(self, client):
        r = client.get(f"/matches/{MATCH_ID}/tracks")
        assert r.status_code == 200
        tracks = r.json()
        assert len(tracks) > 0
        assert "frame_idx" in tracks[0]
        assert "players" in tracks[0]

    def test_tracks_stride(self, client):
        r1 = client.get(f"/matches/{MATCH_ID}/tracks?stride=1")
        r5 = client.get(f"/matches/{MATCH_ID}/tracks?stride=5")
        assert len(r5.json()) < len(r1.json())

    def test_tracks_stride_reduces_size(self, client):
        r = client.get(f"/matches/{MATCH_ID}/tracks?stride=10")
        assert r.status_code == 200
        assert len(r.json()) == 10  # 100 frames / stride 10


class TestAnalytics:
    def test_get_analytics(self, client):
        r = client.get(f"/matches/{MATCH_ID}/analytics")
        assert r.status_code == 200
        data = r.json()
        assert "num_events" in data
        assert "player_stats" in data
        assert len(data["player_stats"]) > 0


class TestNarratives:
    def test_get_narratives_returns_503_without_llm_key(self, client, monkeypatch):
        """Bug-fix 2026-04-10: narrative endpoint now returns 503 instead of
        a hardcoded 'A competitive match between...' summary when no real
        LLM is configured. The fake-LLM fallback would defeat any
        downstream fact-checking or grounding guarantee."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = client.get(f"/matches/{MATCH_ID}/narratives")
        assert r.status_code == 503
        body = r.json()
        # Whatever the body, it must NOT contain the deterministic fake string.
        assert "competitive match between" not in json.dumps(body).lower()

    def test_get_narratives_returns_cached_json_when_present(self, client, tmp_path):
        """When narrative.json is cached on disk it bypasses the LLM call."""
        match_dir = tmp_path / "england_epl" / "2014-2015" / "match_001"
        cached = {
            "match_summary": "cached summary",
            "key_moments": [],
            "player_contributions": [],
            "tactical_breakdown": [],
        }
        (match_dir / "narrative.json").write_text(json.dumps(cached))
        r = client.get(f"/matches/{MATCH_ID}/narratives")
        assert r.status_code == 200
        assert r.json()["match_summary"] == "cached summary"


class TestVideo:
    def test_video_not_found(self, client):
        """Video endpoint returns 404 when file doesn't exist (expected in test)."""
        r = client.get(f"/video/{MATCH_ID}")
        assert r.status_code == 404

    def test_video_range_returns_206(self, client, tmp_path):
        """Bug-fix 2026-04-10: /video/ now parses Range and returns
        206 Partial Content with Content-Range, instead of a plain
        FileResponse with just an Accept-Ranges header (which never
        produces 206 and breaks browser <video> seeking)."""
        match_dir = tmp_path / "england_epl" / "2014-2015" / "match_001"
        video_path = match_dir / "1_720p.mkv"
        video_path.write_bytes(b"x" * 4096)  # 4 KB fake video

        r = client.get(f"/video/{MATCH_ID}", headers={"Range": "bytes=0-1023"})
        assert r.status_code == 206
        assert r.headers["content-range"] == "bytes 0-1023/4096"
        assert r.headers["accept-ranges"] == "bytes"
        assert int(r.headers["content-length"]) == 1024
        assert len(r.content) == 1024

    def test_video_no_range_returns_200(self, client, tmp_path):
        """Without a Range header, /video/ returns the full file."""
        match_dir = tmp_path / "england_epl" / "2014-2015" / "match_001"
        video_path = match_dir / "1_720p.mkv"
        video_path.write_bytes(b"y" * 2048)

        r = client.get(f"/video/{MATCH_ID}")
        assert r.status_code == 200
        assert r.headers["accept-ranges"] == "bytes"
