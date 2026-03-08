import { describe, it, expect } from "vitest";
import { BPMCompatibilityScorer } from "@domain/services/BPMCompatibilityScorer";
import { BPM } from "@domain/value-objects/BPM";

describe("BPMCompatibilityScorer", () => {
  const scorer = new BPMCompatibilityScorer();

  it("should score identical BPMs as 100", () => {
    expect(scorer.score(BPM.create(128), BPM.create(128))).toBe(100);
  });

  it("should score BPMs within +-8 range highly", () => {
    const score = scorer.score(BPM.create(128), BPM.create(132));
    expect(score).toBeGreaterThanOrEqual(70);
  });

  it("should score half-time match highly", () => {
    const score = scorer.score(BPM.create(128), BPM.create(64));
    expect(score).toBeGreaterThanOrEqual(80);
  });

  it("should score double-time match highly", () => {
    const score = scorer.score(BPM.create(85), BPM.create(170));
    expect(score).toBeGreaterThanOrEqual(80);
  });

  it("should score distant BPMs low", () => {
    const score = scorer.score(BPM.create(75), BPM.create(110));
    expect(score).toBeLessThan(30);
  });
});
