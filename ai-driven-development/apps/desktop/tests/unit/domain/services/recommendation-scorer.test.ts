import { describe, it, expect } from "vitest";
import { RecommendationScorer } from "@domain/services/RecommendationScorer";

describe("RecommendationScorer", () => {
  const scorer = new RecommendationScorer();

  it("should produce a weighted composite score", () => {
    const result = scorer.computeScore({
      bpmScore: 100,
      keyScore: 100,
      energyScore: 100,
      genreScore: 100,
      historyScore: 100,
    });
    expect(result).toBe(100);
  });

  it("should apply correct weights (BPM 20%, Key 25%, Energy 25%, Genre 15%, History 15%)", () => {
    const result = scorer.computeScore({
      bpmScore: 100,
      keyScore: 0,
      energyScore: 0,
      genreScore: 0,
      historyScore: 0,
    });
    expect(result).toBe(20);
  });

  it("should apply key weight of 25%", () => {
    const result = scorer.computeScore({
      bpmScore: 0,
      keyScore: 100,
      energyScore: 0,
      genreScore: 0,
      historyScore: 0,
    });
    expect(result).toBe(25);
  });

  it("should apply energy weight of 25%", () => {
    const result = scorer.computeScore({
      bpmScore: 0,
      keyScore: 0,
      energyScore: 100,
      genreScore: 0,
      historyScore: 0,
    });
    expect(result).toBe(25);
  });

  it("should produce ranked list", () => {
    const scores = [
      { bpmScore: 90, keyScore: 80, energyScore: 70, genreScore: 60, historyScore: 50 },
      { bpmScore: 50, keyScore: 60, energyScore: 70, genreScore: 80, historyScore: 90 },
      { bpmScore: 100, keyScore: 100, energyScore: 100, genreScore: 100, historyScore: 100 },
    ];
    const results = scores
      .map((s) => scorer.computeScore(s))
      .sort((a, b) => b - a);
    expect(results[0]).toBe(100);
    expect(results[0]).toBeGreaterThanOrEqual(results[1]!);
    expect(results[1]).toBeGreaterThanOrEqual(results[2]!);
  });

  it("should clamp result between 0 and 100", () => {
    const result = scorer.computeScore({
      bpmScore: 0,
      keyScore: 0,
      energyScore: 0,
      genreScore: 0,
      historyScore: 0,
    });
    expect(result).toBe(0);
  });
});
