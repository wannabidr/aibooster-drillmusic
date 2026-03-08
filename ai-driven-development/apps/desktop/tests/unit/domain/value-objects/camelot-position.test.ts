import { describe, it, expect } from "vitest";
import { CamelotPosition } from "@domain/value-objects/CamelotPosition";

describe("CamelotPosition", () => {
  it("should create from valid notation", () => {
    const pos = CamelotPosition.create("8A");
    expect(pos.number).toBe(8);
    expect(pos.letter).toBe("A");
    expect(pos.toString()).toBe("8A");
  });

  it("should reject invalid notation", () => {
    expect(() => CamelotPosition.create("0A")).toThrow();
    expect(() => CamelotPosition.create("13A")).toThrow();
    expect(() => CamelotPosition.create("8C")).toThrow();
  });

  it("should calculate adjacent keys (same letter +-1)", () => {
    const pos = CamelotPosition.create("8A");
    const adjacent = pos.adjacentKeys();
    const strs = adjacent.map((k) => k.toString());
    expect(strs).toContain("7A");
    expect(strs).toContain("9A");
    expect(strs).toContain("8B"); // mode switch
  });

  it("should wrap around 12 to 1", () => {
    const pos = CamelotPosition.create("12A");
    const adjacent = pos.adjacentKeys();
    const strs = adjacent.map((k) => k.toString());
    expect(strs).toContain("11A");
    expect(strs).toContain("1A");
  });

  it("should wrap around 1 to 12", () => {
    const pos = CamelotPosition.create("1B");
    const adjacent = pos.adjacentKeys();
    const strs = adjacent.map((k) => k.toString());
    expect(strs).toContain("12B");
    expect(strs).toContain("2B");
  });

  it("should score same key as 100", () => {
    const a = CamelotPosition.create("8A");
    const b = CamelotPosition.create("8A");
    expect(a.compatibilityScore(b)).toBe(100);
  });

  it("should score adjacent key (+-1 same letter) as 90", () => {
    const a = CamelotPosition.create("8A");
    const b = CamelotPosition.create("9A");
    expect(a.compatibilityScore(b)).toBe(90);
  });

  it("should score mode switch (same number, different letter) as 85", () => {
    const a = CamelotPosition.create("8A");
    const b = CamelotPosition.create("8B");
    expect(a.compatibilityScore(b)).toBe(85);
  });

  it("should score distant keys low", () => {
    const a = CamelotPosition.create("1A");
    const b = CamelotPosition.create("6B");
    expect(a.compatibilityScore(b)).toBeLessThan(50);
  });

  it("should be equal to same position", () => {
    const a = CamelotPosition.create("5B");
    const b = CamelotPosition.create("5B");
    expect(a.equals(b)).toBe(true);
  });
});
