import { create } from 'zustand';

interface Track {
  title: string;
  artist: string;
  bpm?: number;
  key?: string;
}

interface AppState {
  currentTrack: Track | null;
  recommendations: Track[];
  setCurrentTrack: (track: Track) => void;
  setRecommendations: (tracks: Track[]) => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentTrack: null,
  recommendations: [],
  setCurrentTrack: (track) => set({ currentTrack: track }),
  setRecommendations: (tracks) => set({ recommendations: tracks }),
}));
