/**
 * Tauri IPC invoke wrappers.
 * Provides typed wrappers around Tauri's invoke() for each command.
 * Falls back to mock data in browser (non-Tauri) environments.
 */

import type {
  ImportLibraryResponse,
  TrackDTO,
  AnalysisResultDTO,
  BatchAnalyzeProgress,
  RecommendationDTO,
  PreviewResultDTO,
} from "./commands";

type InvokeFn = (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;

function getInvoke(): InvokeFn | null {
  if (typeof window !== "undefined" && "__TAURI__" in window) {
    // Dynamic import handled at runtime by Tauri
    return (window as Record<string, unknown>).__TAURI_INVOKE__ as InvokeFn;
  }
  return null;
}

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const tauriInvoke = getInvoke();
  if (!tauriInvoke) {
    throw new Error(`Tauri not available. Cannot invoke: ${cmd}`);
  }
  return tauriInvoke(cmd, args) as Promise<T>;
}

export async function importLibrary(
  source: "rekordbox" | "serato" | "traktor" | "folder",
  path: string,
): Promise<ImportLibraryResponse> {
  return invoke<ImportLibraryResponse>("import_library", { source, path });
}

export async function getTracks(params?: {
  offset?: number;
  limit?: number;
  search?: string;
}): Promise<TrackDTO[]> {
  return invoke<TrackDTO[]>("get_tracks", params ?? {});
}

export async function analyzeTrack(
  trackId: string,
  filePath: string,
): Promise<AnalysisResultDTO> {
  return invoke<AnalysisResultDTO>("analyze_track", { trackId, filePath });
}

export async function batchAnalyze(
  trackIds: string[],
): Promise<BatchAnalyzeProgress> {
  return invoke<BatchAnalyzeProgress>("batch_analyze", { trackIds });
}

export async function getRecommendations(
  currentTrackId: string,
  limit?: number,
  excludeTrackIds?: string[],
): Promise<RecommendationDTO[]> {
  return invoke<RecommendationDTO[]>("get_recommendations", {
    currentTrackId,
    limit: limit ?? 10,
    excludeTrackIds: excludeTrackIds ?? [],
  });
}

export async function generatePreview(
  fromTrackId: string,
  toTrackId: string,
  transitionType?: "crossfade" | "cut" | "echo",
  durationMs?: number,
): Promise<PreviewResultDTO> {
  return invoke<PreviewResultDTO>("generate_preview", {
    fromTrackId,
    toTrackId,
    transitionType: transitionType ?? "crossfade",
    durationMs: durationMs ?? 8000,
  });
}
