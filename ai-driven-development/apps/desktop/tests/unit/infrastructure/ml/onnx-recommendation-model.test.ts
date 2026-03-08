import { describe, it, expect, beforeEach } from "vitest";
import {
  ONNXRecommendationModel,
  ONNXSession,
  normalizeFeatures,
  buildPairFeatures,
} from "@infrastructure/ml/ONNXRecommendationModel";
import { TrackAnalysisData } from "@application/dto/RecommendationDTO";

function createMockSession(returnScores: number[]): ONNXSession {
  return {
    async run(feeds) {
      const batchSize = feeds["features"]!.dims[0]!;
      const scores = new Float32Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        scores[i] = returnScores[i] ?? 0.5;
      }
      return { score: { data: scores } };
    },
  };
}

describe("ONNXRecommendationModel", () => {
  let model: ONNXRecommendationModel;

  const trackA: TrackAnalysisData = {
    trackId: "a",
    bpm: 128,
    camelotPosition: "8A",
    energy: 75,
  };

  const trackB: TrackAnalysisData = {
    trackId: "b",
    bpm: 130,
    camelotPosition: "9A",
    energy: 70,
  };

  beforeEach(() => {
    model = new ONNXRecommendationModel();
  });

  it("should report unavailable before loading", () => {
    expect(model.isAvailable).toBe(false);
  });

  it("should report available after loading session", async () => {
    await model.load(createMockSession([0.8]));
    expect(model.isAvailable).toBe(true);
  });

  it("should throw when scoring without loaded session", () => {
    expect(() => model.scorePair(trackA, trackB)).toThrow(
      "ONNX session not loaded",
    );
  });

  it("should score batch async via ONNX session", async () => {
    await model.load(createMockSession([0.85, 0.6]));
    const scores = await model.scoreBatchAsync(trackA, [trackA, trackB]);
    expect(scores).toHaveLength(2);
    expect(scores[0]).toBe(85); // 0.85 * 100 rounded
    expect(scores[1]).toBe(60);
  });

  it("should pass correct feature dimensions to session", async () => {
    let capturedDims: number[] = [];
    const session: ONNXSession = {
      async run(feeds) {
        capturedDims = feeds["features"]!.dims;
        return {
          score: { data: new Float32Array([0.5, 0.5, 0.5]) },
        };
      },
    };
    await model.load(session);
    await model.scoreBatchAsync(trackA, [trackA, trackB, trackA]);
    expect(capturedDims).toEqual([3, 10]);
  });

  it("should produce scores in 0-100 range", async () => {
    const scores = [0.0, 0.25, 0.5, 0.75, 1.0];
    await model.load(createMockSession(scores));
    const candidates = scores.map((_, i) => ({
      ...trackA,
      trackId: `t${i}`,
    }));
    const result = await model.scoreBatchAsync(trackA, candidates);
    result.forEach((s) => {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(100);
    });
  });
});

describe("normalizeFeatures", () => {
  it("should produce 5-element array", () => {
    const features = normalizeFeatures({
      trackId: "t",
      bpm: 128,
      camelotPosition: "8A",
      energy: 75,
    });
    expect(features).toHaveLength(5);
  });

  it("should normalize BPM to roughly 0-1 range", () => {
    const low = normalizeFeatures({
      trackId: "t",
      bpm: 60,
      camelotPosition: "1A",
      energy: 0,
    });
    const high = normalizeFeatures({
      trackId: "t",
      bpm: 200,
      camelotPosition: "1A",
      energy: 100,
    });
    expect(low[0]).toBeCloseTo(0, 1);
    expect(high[0]).toBeCloseTo(1, 1);
  });

  it("should encode key mode as 0 for A and 1 for B", () => {
    const modeA = normalizeFeatures({
      trackId: "t",
      bpm: 128,
      camelotPosition: "5A",
      energy: 50,
    });
    const modeB = normalizeFeatures({
      trackId: "t",
      bpm: 128,
      camelotPosition: "5B",
      energy: 50,
    });
    expect(modeA[2]).toBe(0.0);
    expect(modeB[2]).toBe(1.0);
  });
});

describe("buildPairFeatures", () => {
  it("should produce 10-element Float32Array", () => {
    const pair = buildPairFeatures(
      { trackId: "a", bpm: 128, camelotPosition: "8A", energy: 75 },
      { trackId: "b", bpm: 130, camelotPosition: "9B", energy: 70 },
    );
    expect(pair).toBeInstanceOf(Float32Array);
    expect(pair).toHaveLength(10);
  });
});
