import { describe, it, expect } from "vitest";
import { EnergyProfile } from "@domain/value-objects/EnergyProfile";

describe("EnergyProfile", () => {
  it("should create with valid energy level", () => {
    const profile = EnergyProfile.create({ overall: 75 });
    expect(profile.overall).toBe(75);
  });

  it("should reject energy below 0", () => {
    expect(() => EnergyProfile.create({ overall: -1 })).toThrow();
  });

  it("should reject energy above 100", () => {
    expect(() => EnergyProfile.create({ overall: 101 })).toThrow();
  });

  it("should support optional RMS values", () => {
    const profile = EnergyProfile.create({
      overall: 80,
      rmsValues: [0.3, 0.5, 0.7],
    });
    expect(profile.rmsValues).toEqual([0.3, 0.5, 0.7]);
  });

  it("should determine trajectory as 'build'", () => {
    const profile = EnergyProfile.create({
      overall: 80,
      rmsValues: [0.2, 0.4, 0.6, 0.8],
    });
    expect(profile.trajectory).toBe("build");
  });

  it("should determine trajectory as 'drop'", () => {
    const profile = EnergyProfile.create({
      overall: 40,
      rmsValues: [0.8, 0.6, 0.4, 0.2],
    });
    expect(profile.trajectory).toBe("drop");
  });

  it("should determine trajectory as 'maintain' for flat energy", () => {
    const profile = EnergyProfile.create({
      overall: 60,
      rmsValues: [0.5, 0.5, 0.5, 0.5],
    });
    expect(profile.trajectory).toBe("maintain");
  });

  it("should compute compatibility score", () => {
    const a = EnergyProfile.create({ overall: 75 });
    const b = EnergyProfile.create({ overall: 80 });
    expect(a.compatibilityScore(b)).toBeGreaterThan(80);
  });

  it("should give low score for very different energy levels", () => {
    const a = EnergyProfile.create({ overall: 10 });
    const b = EnergyProfile.create({ overall: 95 });
    expect(a.compatibilityScore(b)).toBeLessThan(30);
  });
});
