/**
 * Tauri IPC command type definitions (TypeScript side).
 * These define the contract for communication with the Rust Tauri backend
 * and the Python analysis sidecar.
 */

// --- Library Management ---

export interface ImportLibraryCommand {
  command: "import_library";
  payload: {
    source: "rekordbox" | "serato" | "traktor" | "folder";
    path: string;
  };
}

export interface ImportLibraryResponse {
  trackCount: number;
  errors: string[];
}

export interface GetTracksCommand {
  command: "get_tracks";
  payload: {
    offset?: number;
    limit?: number;
    search?: string;
  };
}

export interface TrackDTO {
  id: string;
  title: string;
  artist: string;
  album?: string;
  filePath: string;
  durationMs?: number;
  bpm?: number;
  key?: string;
  camelotPosition?: string;
  energy?: number;
  genre?: string;
  fingerprint?: string;
  analyzedAt?: string;
}

// --- Analysis ---

export interface AnalyzeTrackCommand {
  command: "analyze_track";
  payload: {
    trackId: string;
    filePath: string;
  };
}

export interface AnalysisResultDTO {
  trackId: string;
  bpm: number;
  key: string;
  camelotPosition: string;
  energy: number;
  fingerprint?: string;
  spectralCentroid?: number;
  rmsValues?: number[];
}

export interface BatchAnalyzeCommand {
  command: "batch_analyze";
  payload: {
    trackIds: string[];
  };
}

export interface BatchAnalyzeProgress {
  completed: number;
  total: number;
  currentTrackId: string;
  errors: Array<{ trackId: string; error: string }>;
}

// --- Recommendations ---

export interface GetRecommendationsCommand {
  command: "get_recommendations";
  payload: {
    currentTrackId: string;
    limit?: number;
    excludeTrackIds?: string[];
  };
}

export interface RecommendationDTO {
  trackId: string;
  score: number;
  breakdown: {
    bpmScore: number;
    keyScore: number;
    energyScore: number;
    genreScore: number;
    historyScore: number;
  };
  confidence: number;
  reason?: string;
}

// --- Preview ---

export interface GeneratePreviewCommand {
  command: "generate_preview";
  payload: {
    fromTrackId: string;
    toTrackId: string;
    transitionType?: "crossfade" | "cut" | "echo";
    durationMs?: number;
  };
}

export interface PreviewResultDTO {
  audioUrl: string;
  durationMs: number;
  transitionPointMs: number;
}

// --- Event types emitted from Tauri backend ---

export type TauriEvent =
  | { type: "analysis_progress"; data: BatchAnalyzeProgress }
  | { type: "import_progress"; data: { current: number; total: number } }
  | { type: "preview_ready"; data: PreviewResultDTO }
  | { type: "sidecar_error"; data: { message: string } };
