import { describe, it, expect } from "vitest";
import {
  cosineSimilarity,
  computeGenreScore,
} from "@domain/services/GenreSimilarity";

describe("cosineSimilarity", () => {
  it("should return 1.0 for identical vectors", () => {
    const v = [0.1, 0.2, 0.3, 0.4, 0.5];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0, 5);
  });

  it("should return 0.0 for orthogonal vectors", () => {
    const a = [1, 0, 0, 0];
    const b = [0, 1, 0, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
  });

  it("should return -1.0 for opposite vectors", () => {
    const a = [1, 0, 0];
    const b = [-1, 0, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1.0, 5);
  });

  it("should handle 64-dimensional vectors", () => {
    const a = Array.from({ length: 64 }, (_, i) => Math.sin(i));
    const b = Array.from({ length: 64 }, (_, i) => Math.sin(i + 0.1));
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThan(0.9); // very similar shifted sine
    expect(sim).toBeLessThanOrEqual(1.0);
  });

  it("should return 0 when either vector is all zeros", () => {
    const a = [1, 2, 3];
    const b = [0, 0, 0];
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it("should be symmetric", () => {
    const a = [0.3, 0.7, 0.1, 0.9];
    const b = [0.8, 0.2, 0.5, 0.4];
    expect(cosineSimilarity(a, b)).toBeCloseTo(cosineSimilarity(b, a), 10);
  });
});

describe("computeGenreScore", () => {
  it("should return 100 for identical embeddings", () => {
    const v = Array.from({ length: 64 }, (_, i) => Math.sin(i));
    expect(computeGenreScore(v, v)).toBe(100);
  });

  it("should return 50 for orthogonal embeddings", () => {
    const a = [1, 0, 0, 0];
    const b = [0, 1, 0, 0];
    expect(computeGenreScore(a, b)).toBe(50);
  });

  it("should return 0 for opposite embeddings", () => {
    const a = [1, 0, 0];
    const b = [-1, 0, 0];
    expect(computeGenreScore(a, b)).toBe(0);
  });

  it("should return high score for similar genre embeddings (techno vs house)", () => {
    // Simulate similar genres: mostly overlapping embeddings with small perturbation
    const techno = Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.5));
    const house = Array.from(
      { length: 64 },
      (_, i) => Math.sin(i * 0.5) + 0.05 * Math.cos(i),
    );
    const score = computeGenreScore(techno, house);
    expect(score).toBeGreaterThan(80);
  });

  it("should return low score for dissimilar genre embeddings (techno vs classical)", () => {
    // Simulate very different genres
    const techno = Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.5));
    const classical = Array.from({ length: 64 }, (_, i) =>
      Math.cos(i * 3.7 + 2),
    );
    const score = computeGenreScore(techno, classical);
    expect(score).toBeLessThan(60);
  });

  it("should return neutral 50 when either embedding is missing", () => {
    const v = [0.1, 0.2, 0.3];
    expect(computeGenreScore(v, undefined)).toBe(50);
    expect(computeGenreScore(undefined, v)).toBe(50);
    expect(computeGenreScore(undefined, undefined)).toBe(50);
  });
});
