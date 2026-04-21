import React, { useRef, useCallback, useEffect, useState } from 'react';
import { useStore } from '../store';
import { useRealtimeStore } from '../realtimeStore';
import { getVideoUrl } from '../api';
import { LiveTrackingOverlay } from './LiveTrackingOverlay';
import { Play, Film, Radio, Loader } from 'lucide-react';

const WS_BASE = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export const VideoPlayer: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const lastSeekTimeRef = useRef<number>(0);
  const { selectedMatch, setCurrentTime, selectedHalf, setSelectedHalf } = useStore();
  const {
    isConnected, currentFrame, videoInfo,
    showBoundingBoxes, showTrackIds, showConfidence, overlayOpacity,
    setConnected, setProcessing, setCurrentFrame, setVideoInfo, setStats,
    setError, addFrameToTrails, reset,
  } = useRealtimeStore();
  const [realtimeEnabled, setRealtimeEnabled] = useState(false);

  // WebSocket connection management
  useEffect(() => {
    if (!selectedMatch?.has_video || !realtimeEnabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setConnected(false);
      }
      return;
    }

    const encodedId = encodeURI(selectedMatch.id);
    const ws = new WebSocket(`${WS_BASE}/ws/realtime/${encodedId}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setProcessing(false); };
    ws.onerror = () => setError('WebSocket connection failed');

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case 'connected':
            setVideoInfo(msg.video_info);
            break;
          case 'frame_result':
            setCurrentFrame(msg.data);
            addFrameToTrails(msg.data);
            break;
          case 'stats':
            setStats(msg.data);
            break;
          case 'error':
            setError(msg.message);
            break;
        }
      } catch { /* ignore parse errors */ }
    };

    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
      reset();
    };
  }, [selectedMatch, realtimeEnabled, selectedHalf, setConnected, setProcessing, setVideoInfo, setCurrentFrame, setStats, setError, addFrameToTrails, reset]);

  // Update store time on every timeupdate (no WS messages here)
  const handleTimeUpdate = useCallback(() => {
    if (!videoRef.current) return;
    const timeMs = videoRef.current.currentTime * 1000;
    setCurrentTime(timeMs);
  }, [setCurrentTime]);

  // Send seek to WebSocket only on explicit user seeks (>500ms jump)
  const handleSeeked = useCallback(() => {
    if (!videoRef.current) return;
    const timeMs = videoRef.current.currentTime * 1000;
    if (
      wsRef.current?.readyState === WebSocket.OPEN &&
      realtimeEnabled &&
      Math.abs(timeMs - lastSeekTimeRef.current) > 500
    ) {
      lastSeekTimeRef.current = timeMs;
      wsRef.current.send(JSON.stringify({ type: 'seek', timestamp_ms: Math.round(timeMs) }));
    }
  }, [realtimeEnabled]);

  // Request stats periodically when processing
  useEffect(() => {
    if (!isConnected || !realtimeEnabled) return;
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'get_stats' }));
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [isConnected, realtimeEnabled]);

  // Clean up on match change
  useEffect(() => {
    reset();
    setRealtimeEnabled(false);
  }, [selectedMatch, reset]);

  if (!selectedMatch) {
    return (
      <div className="video-player empty">
        <Film size={48} strokeWidth={1} />
        <p>Select a match to view</p>
      </div>
    );
  }

  const toggleRealtime = () => setRealtimeEnabled(!realtimeEnabled);

  return (
    <div className="video-player">
      <div className="video-header">
        <span className="match-title">{selectedMatch.name}</span>
        <div className="video-controls-row">
          {selectedMatch.has_video && (
            <button
              className={`realtime-toggle ${realtimeEnabled ? 'active' : ''}`}
              onClick={toggleRealtime}
              title="Toggle real-time analytics"
            >
              {realtimeEnabled && isConnected ? (
                <><Radio size={14} /> <span>LIVE</span></>
              ) : realtimeEnabled ? (
                <><Loader size={14} className="spin" /> <span>Connecting...</span></>
              ) : (
                <><Radio size={14} /> <span>Analyze</span></>
              )}
            </button>
          )}
          <div className="half-toggle">
            <button className={selectedHalf === 1 ? 'active' : ''} onClick={() => setSelectedHalf(1)}>1st Half</button>
            <button className={selectedHalf === 2 ? 'active' : ''} onClick={() => setSelectedHalf(2)}>2nd Half</button>
          </div>
        </div>
      </div>
      {selectedMatch.has_video ? (
        <div className="video-container" style={{ position: 'relative' }}>
          <video
            ref={videoRef}
            src={getVideoUrl(selectedMatch.id, selectedHalf)}
            controls
            style={{ position: 'relative', zIndex: 1 }}
            onTimeUpdate={handleTimeUpdate}
            onSeeked={handleSeeked}
          />
          {realtimeEnabled && currentFrame && (
            <LiveTrackingOverlay
              players={currentFrame.players}
              videoWidth={videoInfo?.width || 1280}
              videoHeight={videoInfo?.height || 720}
              showBoundingBoxes={showBoundingBoxes}
              showTrackIds={showTrackIds}
              showConfidence={showConfidence}
              overlayOpacity={overlayOpacity}
              isCameraCut={currentFrame.is_camera_cut}
            />
          )}
        </div>
      ) : (
        <div className="no-video">
          <Play size={36} strokeWidth={1} />
          <p>Video not downloaded</p>
          <span>Run: python scripts/download_features.py --video</span>
        </div>
      )}
    </div>
  );
};
