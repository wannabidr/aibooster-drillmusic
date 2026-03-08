export interface PreviewPortResult {
  audioUrl: string;
  durationMs: number;
  transitionPointMs: number;
}

export interface PreviewPort {
  generatePreview(
    fromTrackId: string,
    toTrackId: string,
    transitionType?: string,
    durationMs?: number,
  ): Promise<PreviewPortResult>;
}
