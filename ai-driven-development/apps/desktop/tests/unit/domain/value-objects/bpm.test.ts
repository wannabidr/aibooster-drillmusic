import { describe, it, expect } from "vitest";
import { BPM } from "@domain/value-objects/BPM";

describe("BPM", () => {
  it("should create a valid BPM", () => {
    const bpm = BPM.create(128);
    expect(bpm.value).toBe(128);
  });

  it("should reject BPM below 20", () => {
    expect(() => BPM.create(19)).toThrow();
  });

  it("should reject BPM above 300", () => {
    expect(() => BPM.create(301)).toThrow();
  });

  it("should calculate half time", () => {
    const bpm = BPM.create(128);
    expect(bpm.halfTime().value).toBe(64);
  });

  it("should calculate double time", () => {
    const bpm = BPM.create(85);
    expect(bpm.doubleTime().value).toBe(170);
  });

  it("should allow half time even if below 60", () => {
    const bpm = BPM.create(100);
    // Half time = 50, should still be representable
    expect(bpm.halfTime().value).toBe(50);
  });

  it("should score identical BPMs as 100", () => {
    const a = BPM.create(128);
    const b = BPM.create(128);
    expect(a.compatibilityScore(b)).toBe(100);
  });

  it("should give high score for close BPMs within +-8 range", () => {
    const a = BPM.create(128);
    const b = BPM.create(130);
    expect(a.compatibilityScore(b)).toBeGreaterThan(70);
  });

  it("should give high score for half/double time matches", () => {
    const a = BPM.create(128);
    const b = BPM.create(64);
    expect(a.compatibilityScore(b)).toBeGreaterThan(80);
  });

  it("should give low score for distant BPMs", () => {
    const a = BPM.create(75);
    const b = BPM.create(110);
    expect(a.compatibilityScore(b)).toBeLessThan(30);
  });

  it("should be equal to another BPM with same value", () => {
    const a = BPM.create(128);
    const b = BPM.create(128);
    expect(a.equals(b)).toBe(true);
  });

  it("should not be equal to a different BPM", () => {
    const a = BPM.create(128);
    const b = BPM.create(130);
    expect(a.equals(b)).toBe(false);
  });
});
