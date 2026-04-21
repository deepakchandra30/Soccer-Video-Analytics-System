import { create } from 'zustand';

export interface PlayerDetection {
  track_id: number;
  bbox: [number, number, number, number];
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

// Track player positions over time for trail rendering
export interface PlayerTrail {
  track_id: number;
  positions: Array<{ x: number; y: number; timestamp_ms: number }>;
}

interface RealtimeState {
  // Connection state
  isConnected: boolean;
  isProcessing: boolean;
  error: string | null;

  // Video info
  videoInfo: VideoInfo | null;

  // Current frame data
  currentFrame: FrameResult | null;

  // Running stats
  stats: RealtimeStats | null;

  // Player trails (last 100 positions per player)
  playerTrails: Map<number, PlayerTrail>;

  // Overlay settings
  showBoundingBoxes: boolean;
  showTrackIds: boolean;
  showConfidence: boolean;
  showTrails: boolean;
  overlayOpacity: number;

  // Actions
  setConnected: (connected: boolean) => void;
  setProcessing: (processing: boolean) => void;
  setError: (error: string | null) => void;
  setVideoInfo: (info: VideoInfo | null) => void;
  setCurrentFrame: (frame: FrameResult | null) => void;
  setStats: (stats: RealtimeStats | null) => void;
  addFrameToTrails: (frame: FrameResult) => void;
  toggleBoundingBoxes: () => void;
  toggleTrackIds: () => void;
  toggleConfidence: () => void;
  toggleTrails: () => void;
  setOverlayOpacity: (opacity: number) => void;
  reset: () => void;
}

const MAX_TRAIL_LENGTH = 100;

export const useRealtimeStore = create<RealtimeState>((set) => ({
  // Connection state
  isConnected: false,
  isProcessing: false,
  error: null,

  // Video info
  videoInfo: null,

  // Current frame data
  currentFrame: null,

  // Running stats
  stats: null,

  // Player trails
  playerTrails: new Map(),

  // Overlay settings
  showBoundingBoxes: true,
  showTrackIds: true,
  showConfidence: false,
  showTrails: false,
  overlayOpacity: 0.8,

  // Actions
  setConnected: (connected) => set({ isConnected: connected }),
  setProcessing: (processing) => set({ isProcessing: processing }),
  setError: (error) => set({ error }),
  setVideoInfo: (info) => set({ videoInfo: info }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setStats: (stats) => set({ stats }),

  addFrameToTrails: (frame) =>
    set((state) => {
      const newTrails = new Map(state.playerTrails);

      for (const player of frame.players) {
        const x = (player.bbox[0] + player.bbox[2]) / 2;
        const y = (player.bbox[1] + player.bbox[3]) / 2;

        const existing = newTrails.get(player.track_id);
        const newPosition = { x, y, timestamp_ms: frame.timestamp_ms };

        if (existing) {
          const positions = [...existing.positions, newPosition];
          if (positions.length > MAX_TRAIL_LENGTH) {
            positions.splice(0, positions.length - MAX_TRAIL_LENGTH);
          }
          newTrails.set(player.track_id, {
            track_id: player.track_id,
            positions,
          });
        } else {
          newTrails.set(player.track_id, {
            track_id: player.track_id,
            positions: [newPosition],
          });
        }
      }

      return { playerTrails: newTrails };
    }),

  toggleBoundingBoxes: () =>
    set((state) => ({ showBoundingBoxes: !state.showBoundingBoxes })),
  toggleTrackIds: () =>
    set((state) => ({ showTrackIds: !state.showTrackIds })),
  toggleConfidence: () =>
    set((state) => ({ showConfidence: !state.showConfidence })),
  toggleTrails: () =>
    set((state) => ({ showTrails: !state.showTrails })),
  setOverlayOpacity: (opacity) => set({ overlayOpacity: opacity }),

  reset: () =>
    set({
      isConnected: false,
      isProcessing: false,
      error: null,
      videoInfo: null,
      currentFrame: null,
      stats: null,
      playerTrails: new Map(),
    }),
}));
