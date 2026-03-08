import type { PreviewPort } from "@application/ports/PreviewPort";

export interface PreviewRequest {
  fromTrackId: string;
  toTrackId: string;
  transitionType?: "crossfade" | "cut" | "echo";
  durationMs?: number;
}

export interface PreviewResult {
  audioUrl: string;
  durationMs: number;
  transitionPointMs: number;
}

export class GeneratePreview {
  private cache = new Map<string, PreviewResult>();

  constructor(private readonly previewPort: PreviewPort) {}

  async execute(request: PreviewRequest): Promise<PreviewResult> {
    const cacheKey = `${request.fromTrackId}-${request.toTrackId}-${request.transitionType ?? "crossfade"}`;

    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const result = await this.previewPort.generatePreview(
      request.fromTrackId,
      request.toTrackId,
      request.transitionType,
      request.durationMs,
    );

    const preview: PreviewResult = {
      audioUrl: result.audioUrl,
      durationMs: result.durationMs,
      transitionPointMs: result.transitionPointMs,
    };

    this.cache.set(cacheKey, preview);
    return preview;
  }

  async preRenderTopN(
    currentTrackId: string,
    recommendedTrackIds: string[],
    count = 3,
  ): Promise<void> {
    const toPreRender = recommendedTrackIds.slice(0, count);
    await Promise.allSettled(
      toPreRender.map((trackId) =>
        this.execute({ fromTrackId: currentTrackId, toTrackId: trackId }),
      ),
    );
  }

  clearCache(): void {
    this.cache.clear();
  }
}
