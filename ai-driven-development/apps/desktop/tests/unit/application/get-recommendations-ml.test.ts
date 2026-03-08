import { describe, it, expect, beforeEach } from "vitest";
import { GetRecommendations } from "@application/use-cases/GetRecommendations";
import { InMemoryAnalysisDataProvider } from "@infrastructure/repositories/InMemoryAnalysisDataProvider";
import { ScoringPort } from "@application/ports/ScoringPort";
import { TrackAnalysisData } from "@application/dto/RecommendationDTO";

const currentTrack: TrackAnalysisData = {
  trackId: "current",
  bpm: 128,
  camelotPosition: "8A",
  energy: 75,
};

const trackA: TrackAnalysisData = {
  trackId: "a",
  bpm: 130,
  camelotPosition: "9A",
  energy: 70,
};

const trackB: TrackAnalysisData = {
  trackId: "b",
  bpm: 75,
  camelotPosition: "1B",
  energy: 20,
};

function createMockMLScorer(scores: Map<string, number>): ScoringPort {
  return {
    isAvailable: true,
    scorePair(_current, candidate) {
      return scores.get(candidate.trackId) ?? 50;
    },
    scoreBatch(_current, candidates) {
      return candidates.map((c) => scores.get(c.trackId) ?? 50);
    },
  };
}

function createUnavailableMLScorer(): ScoringPort {
  return {
    isAvailable: false,
    scorePair() {
      throw new Error("ML not available");
    },
    scoreBatch() {
      throw new Error("ML not available");
    },
  };
}

describe("GetRecommendations with ML scoring", () => {
  let provider: InMemoryAnalysisDataProvider;

  beforeEach(() => {
    provider = new InMemoryAnalysisDataProvider();
    provider.add(currentTrack);
    provider.add(trackA);
    provider.add(trackB);
  });

  it("should use ML scores when ML scorer is available", () => {
    const mlScores = new Map([
      ["a", 90],
      ["b", 80],
    ]);
    const useCase = new GetRecommendations(
      provider,
      createMockMLScorer(mlScores),
    );
    const results = useCase.execute({ currentTrackId: "current" });
    expect(results[0]!.trackId).toBe("a");
    expect(results[0]!.score).toBe(90);
    expect(results[1]!.trackId).toBe("b");
    expect(results[1]!.score).toBe(80);
  });

  it("should reverse ranking when ML scores differ from rule-based", () => {
    // ML says trackB is better than trackA (opposite of rule-based)
    const mlScores = new Map([
      ["a", 30],
      ["b", 95],
    ]);
    const useCase = new GetRecommendations(
      provider,
      createMockMLScorer(mlScores),
    );
    const results = useCase.execute({ currentTrackId: "current" });
    expect(results[0]!.trackId).toBe("b");
    expect(results[0]!.score).toBe(95);
  });

  it("should fall back to rule-based when ML scorer is unavailable", () => {
    const useCase = new GetRecommendations(
      provider,
      createUnavailableMLScorer(),
    );
    const results = useCase.execute({ currentTrackId: "current" });
    // Should still work with rule-based scoring
    expect(results.length).toBe(2);
    // Rule-based: trackA (similar BPM/key) should score higher than trackB
    expect(results[0]!.trackId).toBe("a");
  });

  it("should fall back to rule-based when no ML scorer provided", () => {
    const useCase = new GetRecommendations(provider);
    const results = useCase.execute({ currentTrackId: "current" });
    expect(results.length).toBe(2);
    expect(results[0]!.trackId).toBe("a");
  });

  it("should respect limit with ML scoring", () => {
    const mlScores = new Map([
      ["a", 90],
      ["b", 80],
    ]);
    const useCase = new GetRecommendations(
      provider,
      createMockMLScorer(mlScores),
    );
    const results = useCase.execute({ currentTrackId: "current", limit: 1 });
    expect(results).toHaveLength(1);
    expect(results[0]!.trackId).toBe("a");
  });

  it("should exclude tracks with ML scoring", () => {
    const mlScores = new Map([
      ["a", 90],
      ["b", 80],
    ]);
    const useCase = new GetRecommendations(
      provider,
      createMockMLScorer(mlScores),
    );
    const results = useCase.execute({
      currentTrackId: "current",
      excludeTrackIds: ["a"],
    });
    expect(results).toHaveLength(1);
    expect(results[0]!.trackId).toBe("b");
  });
});
