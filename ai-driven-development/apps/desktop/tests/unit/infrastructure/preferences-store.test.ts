import { describe, it, expect, beforeEach } from "vitest";
import { usePreferencesStore } from "@infrastructure/state/usePreferencesStore";

describe("usePreferencesStore", () => {
  beforeEach(() => {
    usePreferencesStore.getState().reset();
  });

  it("should have default weights matching PRD spec (BPM 20%, Key 25%, Energy 25%, Genre 15%, History 15%)", () => {
    const { weights } = usePreferencesStore.getState();
    expect(weights.bpm).toBe(20);
    expect(weights.key).toBe(25);
    expect(weights.energy).toBe(25);
    expect(weights.genre).toBe(15);
    expect(weights.history).toBe(15);
  });

  it("should default to maintain energy direction", () => {
    expect(usePreferencesStore.getState().energyDirection).toBe("maintain");
  });

  it("should default to long_blend blend style", () => {
    expect(usePreferencesStore.getState().blendStyle).toBe("long_blend");
  });

  it("should update a specific weight", () => {
    usePreferencesStore.getState().setWeight("bpm", 35);
    expect(usePreferencesStore.getState().weights.bpm).toBe(35);
  });

  it("should clamp weights to 0-100", () => {
    usePreferencesStore.getState().setWeight("bpm", 150);
    expect(usePreferencesStore.getState().weights.bpm).toBe(100);
    usePreferencesStore.getState().setWeight("bpm", -10);
    expect(usePreferencesStore.getState().weights.bpm).toBe(0);
  });

  it("should update energy direction", () => {
    usePreferencesStore.getState().setEnergyDirection("build");
    expect(usePreferencesStore.getState().energyDirection).toBe("build");
    usePreferencesStore.getState().setEnergyDirection("drop");
    expect(usePreferencesStore.getState().energyDirection).toBe("drop");
  });

  it("should update blend style", () => {
    usePreferencesStore.getState().setBlendStyle("echo_out");
    expect(usePreferencesStore.getState().blendStyle).toBe("echo_out");
  });

  it("should reset to defaults", () => {
    usePreferencesStore.getState().setWeight("bpm", 50);
    usePreferencesStore.getState().setEnergyDirection("build");
    usePreferencesStore.getState().setBlendStyle("backspin");
    usePreferencesStore.getState().reset();

    const state = usePreferencesStore.getState();
    expect(state.weights.bpm).toBe(20);
    expect(state.energyDirection).toBe("maintain");
    expect(state.blendStyle).toBe("long_blend");
  });

  it("should compute normalized weights (sum to 1.0)", () => {
    const normalized = usePreferencesStore.getState().getNormalizedWeights();
    const sum = normalized.bpm + normalized.key + normalized.energy + normalized.genre + normalized.history;
    expect(Math.abs(sum - 1.0)).toBeLessThan(0.001);
  });

  it("should handle zero weights gracefully in normalization", () => {
    usePreferencesStore.getState().setWeight("bpm", 0);
    usePreferencesStore.getState().setWeight("key", 0);
    usePreferencesStore.getState().setWeight("energy", 0);
    usePreferencesStore.getState().setWeight("genre", 0);
    usePreferencesStore.getState().setWeight("history", 0);
    const normalized = usePreferencesStore.getState().getNormalizedWeights();
    expect(normalized.bpm).toBe(0.2);
    expect(normalized.key).toBe(0.2);
    expect(normalized.energy).toBe(0.2);
    expect(normalized.genre).toBe(0.2);
    expect(normalized.history).toBe(0.2);
  });
});
