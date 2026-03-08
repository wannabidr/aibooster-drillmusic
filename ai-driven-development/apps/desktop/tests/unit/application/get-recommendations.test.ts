import { describe, it, expect, beforeEach } from "vitest";
import { GetRecommendations } from "@application/use-cases/GetRecommendations";
import { InMemoryAnalysisDataProvider } from "@infrastructure/repositories/InMemoryAnalysisDataProvider";
import { TrackAnalysisData } from "@application/dto/RecommendationDTO";

describe("GetRecommendations", () => {
  let provider: InMemoryAnalysisDataProvider;
  let useCase: GetRecommendations;

  const currentTrack: TrackAnalysisData = {
    trackId: "current",
    bpm: 128,
    camelotPosition: "8A",
    energy: 75,
  };

  const perfectMatch: TrackAnalysisData = {
    trackId: "perfect",
    bpm: 128,
    camelotPosition: "8A",
    energy: 75,
  };

  const goodMatch: TrackAnalysisData = {
    trackId: "good",
    bpm: 130,
    camelotPosition: "9A",
    energy: 70,
  };

  const poorMatch: TrackAnalysisData = {
    trackId: "poor",
    bpm: 75,
    camelotPosition: "1B",
    energy: 20,
  };

  beforeEach(() => {
    provider = new InMemoryAnalysisDataProvider();
    useCase = new GetRecommendations(provider);

    provider.add(currentTrack);
    provider.add(perfectMatch);
    provider.add(goodMatch);
    provider.add(poorMatch);
  });

  it("should return recommendations ranked by score", () => {
    const results = useCase.execute({ currentTrackId: "current" });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]!.trackId).toBe("perfect");
    // Verify sorted descending
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1]!.score).toBeGreaterThanOrEqual(results[i]!.score);
    }
  });

  it("should exclude current track from results", () => {
    const results = useCase.execute({ currentTrackId: "current" });
    expect(results.find((r) => r.trackId === "current")).toBeUndefined();
  });

  it("should exclude specified track IDs", () => {
    const results = useCase.execute({
      currentTrackId: "current",
      excludeTrackIds: ["perfect"],
    });
    expect(results.find((r) => r.trackId === "perfect")).toBeUndefined();
  });

  it("should respect limit parameter", () => {
    const results = useCase.execute({ currentTrackId: "current", limit: 1 });
    expect(results.length).toBe(1);
  });

  it("should return empty array if current track not found", () => {
    const results = useCase.execute({ currentTrackId: "nonexistent" });
    expect(results).toEqual([]);
  });

  it("should include score breakdown in results", () => {
    const results = useCase.execute({ currentTrackId: "current" });
    const result = results[0]!;
    expect(result.breakdown).toBeDefined();
    expect(result.breakdown.bpmScore).toBeGreaterThanOrEqual(0);
    expect(result.breakdown.keyScore).toBeGreaterThanOrEqual(0);
    expect(result.breakdown.energyScore).toBeGreaterThanOrEqual(0);
    expect(result.breakdown.genreScore).toBe(50); // neutral
    expect(result.breakdown.historyScore).toBe(50); // neutral
  });

  it("should include confidence score", () => {
    const results = useCase.execute({ currentTrackId: "current" });
    expect(results[0]!.confidence).toBeGreaterThanOrEqual(0);
    expect(results[0]!.confidence).toBeLessThanOrEqual(100);
  });

  it("should score perfect BPM/key/energy match highest", () => {
    const results = useCase.execute({ currentTrackId: "current" });
    const perfect = results.find((r) => r.trackId === "perfect")!;
    const poor = results.find((r) => r.trackId === "poor")!;
    expect(perfect.score).toBeGreaterThan(poor.score);
  });

  it("should complete within 200ms for 100 candidates", () => {
    // Add 100 tracks
    for (let i = 0; i < 100; i++) {
      provider.add({
        trackId: `track-${i}`,
        bpm: 60 + (i % 140),
        camelotPosition: `${(i % 12) + 1}${i % 2 === 0 ? "A" : "B"}`,
        energy: i % 100,
      });
    }

    const start = performance.now();
    const results = useCase.execute({ currentTrackId: "current", limit: 10 });
    const elapsed = performance.now() - start;

    expect(results.length).toBe(10);
    expect(elapsed).toBeLessThan(200);
  });
});
