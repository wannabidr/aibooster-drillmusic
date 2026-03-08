import { describe, it, expect } from "vitest";
import { HarmonicCompatibilityCalculator } from "@domain/services/HarmonicCompatibilityCalculator";
import { CamelotPosition } from "@domain/value-objects/CamelotPosition";

describe("HarmonicCompatibilityCalculator", () => {
  const calc = new HarmonicCompatibilityCalculator();

  it("should score same key as 100", () => {
    const a = CamelotPosition.create("8A");
    const b = CamelotPosition.create("8A");
    expect(calc.score(a, b)).toBe(100);
  });

  it("should score +-1 same letter as 90", () => {
    const a = CamelotPosition.create("8A");
    expect(calc.score(a, CamelotPosition.create("7A"))).toBe(90);
    expect(calc.score(a, CamelotPosition.create("9A"))).toBe(90);
  });

  it("should score mode switch (energy boost) as 85", () => {
    const a = CamelotPosition.create("8A");
    const b = CamelotPosition.create("8B");
    expect(calc.score(a, b)).toBe(85);
  });

  it("should handle wrapping 12 -> 1", () => {
    const a = CamelotPosition.create("12B");
    const b = CamelotPosition.create("1B");
    expect(calc.score(a, b)).toBe(90);
  });
});
