import { describe, it, expect, vi, beforeEach } from "vitest";
import { GeneratePreview } from "@application/use-cases/GeneratePreview";
import type { PreviewPort } from "@application/ports/PreviewPort";

function createMockPreviewPort(): PreviewPort {
  return {
    generatePreview: vi.fn().mockResolvedValue({
      audioUrl: "blob:mock-audio-url",
      durationMs: 16000,
      transitionPointMs: 8000,
    }),
  };
}

describe("GeneratePreview", () => {
  let useCase: GeneratePreview;
  let mockPort: PreviewPort;

  beforeEach(() => {
    mockPort = createMockPreviewPort();
    useCase = new GeneratePreview(mockPort);
    vi.clearAllMocks();
  });

  it("should generate a preview for two tracks", async () => {
    const result = await useCase.execute({
      fromTrackId: "track-a",
      toTrackId: "track-b",
    });
    expect(result.audioUrl).toBe("blob:mock-audio-url");
    expect(result.durationMs).toBe(16000);
    expect(result.transitionPointMs).toBe(8000);
  });

  it("should cache previews and return cached result on second call", async () => {
    await useCase.execute({ fromTrackId: "a", toTrackId: "b" });
    await useCase.execute({ fromTrackId: "a", toTrackId: "b" });

    expect(mockPort.generatePreview).toHaveBeenCalledTimes(1);
  });

  it("should not cache different track pairs", async () => {
    await useCase.execute({ fromTrackId: "a", toTrackId: "b" });
    await useCase.execute({ fromTrackId: "a", toTrackId: "c" });

    expect(mockPort.generatePreview).toHaveBeenCalledTimes(2);
  });

  it("should clear cache", async () => {
    await useCase.execute({ fromTrackId: "a", toTrackId: "b" });
    useCase.clearCache();
    await useCase.execute({ fromTrackId: "a", toTrackId: "b" });

    expect(mockPort.generatePreview).toHaveBeenCalledTimes(2);
  });

  it("should pre-render top N recommendations", async () => {
    await useCase.preRenderTopN("current", ["r1", "r2", "r3", "r4", "r5"], 3);

    expect(mockPort.generatePreview).toHaveBeenCalledTimes(3);
  });
});
