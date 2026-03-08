/**
 * Tauri event listener wrappers.
 * Provides typed wrappers for listening to events from the Rust backend.
 */

import type { BatchAnalyzeProgress, PreviewResultDTO } from "./commands";

type UnlistenFn = () => void;
type EventCallback<T> = (payload: T) => void;

interface TauriEvent<T> {
  payload: T;
}

type ListenFn = (
  event: string,
  handler: (event: TauriEvent<unknown>) => void,
) => Promise<UnlistenFn>;

function getListen(): ListenFn | null {
  if (typeof window !== "undefined" && "__TAURI__" in window) {
    return (window as Record<string, unknown>).__TAURI_LISTEN__ as ListenFn;
  }
  return null;
}

async function listen<T>(
  event: string,
  callback: EventCallback<T>,
): Promise<UnlistenFn> {
  const tauriListen = getListen();
  if (!tauriListen) {
    console.warn(`Tauri not available. Cannot listen to: ${event}`);
    return () => {};
  }
  return tauriListen(event, (e) => callback(e.payload as T));
}

export async function onAnalysisProgress(
  callback: EventCallback<BatchAnalyzeProgress>,
): Promise<UnlistenFn> {
  return listen("analysis_progress", callback);
}

export async function onImportProgress(
  callback: EventCallback<{ current: number; total: number }>,
): Promise<UnlistenFn> {
  return listen("import_progress", callback);
}

export async function onPreviewReady(
  callback: EventCallback<PreviewResultDTO>,
): Promise<UnlistenFn> {
  return listen("preview_ready", callback);
}

export async function onSidecarError(
  callback: EventCallback<{ message: string }>,
): Promise<UnlistenFn> {
  return listen("sidecar_error", callback);
}
