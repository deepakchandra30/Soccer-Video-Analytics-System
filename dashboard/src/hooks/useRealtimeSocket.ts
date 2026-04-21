import { useCallback, useEffect, useRef, useState } from 'react';

const WS_BASE = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export interface PlayerDetection {
  track_id: number;
  bbox: [number, number, number, number]; // x1, y1, x2, y2
  confidence: number;
  pitch_xy: [number, number] | null;
}

export interface FrameResult {
  frame_idx: number;
  timestamp_ms: number;
  is_camera_cut: boolean;
  num_players: number;
  players: PlayerDetection[];
}

export interface RealtimeStats {
  frames_processed: number;
  total_players_detected: number;
  avg_players_per_frame: number;
  avg_confidence: number;
  camera_cuts_detected: number;
  unique_track_ids: number;
  processing_fps: number;
}

export interface VideoInfo {
  fps: number;
  total_frames: number;
  width: number;
  height: number;
  duration_ms: number;
}

interface UseRealtimeSocketReturn {
  isConnected: boolean;
  isProcessing: boolean;
  currentFrame: FrameResult | null;
  stats: RealtimeStats | null;
  videoInfo: VideoInfo | null;
  error: string | null;
  connect: (matchId: string) => void;
  disconnect: () => void;
  seekTo: (timestampMs: number) => void;
  play: (startMs?: number, speed?: number) => void;
  pause: () => void;
  requestStats: () => void;
}

const MAX_RECONNECT_RETRIES = 3;
const RECONNECT_DELAY_MS = 3000;

function useRealtimeSocket(): UseRealtimeSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const matchIdRef = useRef<string | null>(null);
  const reconnectCountRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalCloseRef = useRef(false);

  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<FrameResult | null>(null);
  const [stats, setStats] = useState<RealtimeStats | null>(null);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const connectSocket = useCallback((matchId: string) => {
    // Close any existing connection
    if (wsRef.current) {
      intentionalCloseRef.current = true;
      wsRef.current.close();
      wsRef.current = null;
    }

    clearReconnectTimer();
    intentionalCloseRef.current = false;
    matchIdRef.current = matchId;
    setError(null);

    const url = `${WS_BASE}/ws/realtime/${matchId}`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      reconnectCountRef.current = 0;
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);

        switch (message.type) {
          case 'connected':
            setVideoInfo(message.video_info ?? null);
            setIsConnected(true);
            setError(null);
            break;
          case 'frame_result':
            setCurrentFrame(message as FrameResult);
            break;
          case 'stats':
            setStats(message.stats ?? (message as RealtimeStats));
            break;
          case 'error':
            setError(message.message ?? message.error ?? 'Unknown error');
            break;
          default:
            break;
        }
      } catch {
        // Ignore non-JSON messages
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      setIsConnected(false);
      setIsProcessing(false);

      if (
        !intentionalCloseRef.current &&
        matchIdRef.current &&
        reconnectCountRef.current < MAX_RECONNECT_RETRIES
      ) {
        reconnectCountRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          if (matchIdRef.current) {
            connectSocket(matchIdRef.current);
          }
        }, RECONNECT_DELAY_MS);
      }
    };

    wsRef.current = ws;
  }, [clearReconnectTimer]);

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    clearReconnectTimer();
    matchIdRef.current = null;
    reconnectCountRef.current = 0;

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsProcessing(false);
    setCurrentFrame(null);
    setStats(null);
    setVideoInfo(null);
    setError(null);
  }, [clearReconnectTimer]);

  const seekTo = useCallback(
    (timestampMs: number) => {
      sendMessage({ type: 'seek', timestamp_ms: timestampMs });
    },
    [sendMessage],
  );

  const play = useCallback(
    (startMs?: number, speed?: number) => {
      sendMessage({ type: 'play', start_ms: startMs, speed });
      setIsProcessing(true);
    },
    [sendMessage],
  );

  const pause = useCallback(() => {
    sendMessage({ type: 'pause' });
    setIsProcessing(false);
  }, [sendMessage]);

  const requestStats = useCallback(() => {
    sendMessage({ type: 'get_stats' });
  }, [sendMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      intentionalCloseRef.current = true;
      clearReconnectTimer();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [clearReconnectTimer]);

  return {
    isConnected,
    isProcessing,
    currentFrame,
    stats,
    videoInfo,
    error,
    connect: connectSocket,
    disconnect,
    seekTo,
    play,
    pause,
    requestStats,
  };
}

export { useRealtimeSocket };
export default useRealtimeSocket;
