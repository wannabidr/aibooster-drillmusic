import { create } from 'zustand';

interface Track {
  title: string;
  artist: string;
  bpm?: number;
  key?: string;
  filePath?: string;
}

interface Recommendation {
  track_id: string;
  path: string;
  title?: string;
  artist?: string;
  score: number;
  bpm?: number;
  key?: string;
}

interface AppState {
  currentTrack: Track | null;
  recommendations: Recommendation[];
  isLoadingRecommendations: boolean;
  setCurrentTrack: (track: Track | null) => void;
  setRecommendations: (recs: Recommendation[]) => void;
  setLoadingRecommendations: (loading: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentTrack: null,
  recommendations: [],
  isLoadingRecommendations: false,
  setCurrentTrack: (track) => set({ currentTrack: track }),
  setRecommendations: (recs) => set({ recommendations: recs }),
  setLoadingRecommendations: (loading) => set({ isLoadingRecommendations: loading }),
}));
