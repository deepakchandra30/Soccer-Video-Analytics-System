import { create } from 'zustand';

export interface Match {
  id: string;
  name: string;
  league: string;
  season: string;
  date?: string;
  has_features?: boolean;
  has_video?: boolean;
}

export interface MatchEvent {
  label: string;
  half: number;
  position: number;
  confidence: number;
  game_time?: string;
  team?: string;
  visibility?: string;
}

export interface Analytics {
  total_events: number;
  event_counts: Record<string, number>;
  team_events: Record<string, Record<string, number>>;
  half1_events: number;
  half2_events: number;
  visible_events: number;
  feature_stats: {
    half1_frames: number;
    half2_frames: number;
    total_frames: number;
    feat_dim: number;
    duration_min: number;
  };
}

interface AppState {
  matches: Match[];
  selectedMatch: Match | null;
  events: MatchEvent[];
  analytics: Analytics | null;
  currentTime: number;
  selectedHalf: number;
  loading: boolean;
  searchQuery: string;

  setMatches: (m: Match[]) => void;
  selectMatch: (m: Match | null) => void;
  setEvents: (e: MatchEvent[]) => void;
  setAnalytics: (a: Analytics) => void;
  setCurrentTime: (ms: number) => void;
  setSelectedHalf: (h: number) => void;
  setLoading: (l: boolean) => void;
  setSearchQuery: (q: string) => void;
}

export const useStore = create<AppState>((set) => ({
  matches: [],
  selectedMatch: null,
  events: [],
  analytics: null,
  currentTime: 0,
  selectedHalf: 1,
  loading: false,
  searchQuery: '',

  setMatches: (matches) => set({ matches }),
  selectMatch: (match) => set({ selectedMatch: match, events: [], analytics: null }),
  setEvents: (events) => set({ events }),
  setAnalytics: (analytics) => set({ analytics }),
  setCurrentTime: (ms) => set({ currentTime: ms }),
  setSelectedHalf: (h) => set({ selectedHalf: h }),
  setLoading: (loading) => set({ loading }),
  setSearchQuery: (searchQuery) => set({ searchQuery }),
}));
