import { create } from "zustand";

interface PlayerState {
  currentTrackId: string | null;
  title: string | null;
  artist: string | null;
  bpm: number | null;
  camelotKey: string | null;
  energy: number | null;
  durationMs: number | null;
  elapsedMs: number;
  isPlaying: boolean;
  isPreviewPlaying: boolean;
  previewTrackId: string | null;

  setCurrentTrack: (track: {
    id: string;
    title: string;
    artist: string;
    bpm?: number;
    camelotKey?: string;
    energy?: number;
    durationMs?: number;
  }) => void;
  setElapsed: (ms: number) => void;
  setPlaying: (playing: boolean) => void;
  startPreview: (trackId: string) => void;
  stopPreview: () => void;
  clearTrack: () => void;
}

export const usePlayerStore = create<PlayerState>((set) => ({
  currentTrackId: null,
  title: null,
  artist: null,
  bpm: null,
  camelotKey: null,
  energy: null,
  durationMs: null,
  elapsedMs: 0,
  isPlaying: false,
  isPreviewPlaying: false,
  previewTrackId: null,

  setCurrentTrack: (track) =>
    set({
      currentTrackId: track.id,
      title: track.title,
      artist: track.artist,
      bpm: track.bpm ?? null,
      camelotKey: track.camelotKey ?? null,
      energy: track.energy ?? null,
      durationMs: track.durationMs ?? null,
      elapsedMs: 0,
      isPlaying: true,
    }),
  setElapsed: (ms) => set({ elapsedMs: ms }),
  setPlaying: (playing) => set({ isPlaying: playing }),
  startPreview: (trackId) => set({ isPreviewPlaying: true, previewTrackId: trackId }),
  stopPreview: () => set({ isPreviewPlaying: false, previewTrackId: null }),
  clearTrack: () =>
    set({
      currentTrackId: null,
      title: null,
      artist: null,
      bpm: null,
      camelotKey: null,
      energy: null,
      durationMs: null,
      elapsedMs: 0,
      isPlaying: false,
    }),
}));
