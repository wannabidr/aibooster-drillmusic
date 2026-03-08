import { create } from "zustand";

export interface LibraryTrack {
  id: string;
  title: string;
  artist: string;
  album?: string;
  filePath: string;
  durationMs?: number;
  bpm?: number;
  camelotKey?: string;
  energy?: number;
  genre?: string;
  analyzed: boolean;
}

type ImportSource = "rekordbox" | "serato" | "traktor" | "folder";

interface ImportProgress {
  source: ImportSource;
  current: number;
  total: number;
}

interface LibraryState {
  tracks: LibraryTrack[];
  selectedTrackId: string | null;
  isImporting: boolean;
  importProgress: ImportProgress | null;
  searchQuery: string;

  setTracks: (tracks: LibraryTrack[]) => void;
  addTracks: (tracks: LibraryTrack[]) => void;
  updateTrack: (id: string, updates: Partial<LibraryTrack>) => void;
  selectTrack: (id: string | null) => void;
  setImporting: (importing: boolean) => void;
  setImportProgress: (progress: ImportProgress | null) => void;
  setSearchQuery: (query: string) => void;
}

export const useLibraryStore = create<LibraryState>((set) => ({
  tracks: [],
  selectedTrackId: null,
  isImporting: false,
  importProgress: null,
  searchQuery: "",

  setTracks: (tracks) => set({ tracks }),
  addTracks: (newTracks) =>
    set((state) => ({ tracks: [...state.tracks, ...newTracks] })),
  updateTrack: (id, updates) =>
    set((state) => ({
      tracks: state.tracks.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    })),
  selectTrack: (id) => set({ selectedTrackId: id }),
  setImporting: (importing) => set({ isImporting: importing }),
  setImportProgress: (progress) => set({ importProgress: progress }),
  setSearchQuery: (query) => set({ searchQuery: query }),
}));
