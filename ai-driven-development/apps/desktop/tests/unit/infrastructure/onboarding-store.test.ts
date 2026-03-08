import { describe, it, expect, beforeEach } from "vitest";
import { useOnboardingStore } from "@infrastructure/state/useOnboardingStore";

describe("useOnboardingStore", () => {
  beforeEach(() => {
    useOnboardingStore.getState().reset();
  });

  it("should start at step 0 (welcome)", () => {
    expect(useOnboardingStore.getState().currentStep).toBe(0);
  });

  it("should not be completed initially", () => {
    expect(useOnboardingStore.getState().completed).toBe(false);
  });

  it("should advance to next step", () => {
    useOnboardingStore.getState().nextStep();
    expect(useOnboardingStore.getState().currentStep).toBe(1);
  });

  it("should go back to previous step", () => {
    useOnboardingStore.getState().nextStep();
    useOnboardingStore.getState().nextStep();
    useOnboardingStore.getState().prevStep();
    expect(useOnboardingStore.getState().currentStep).toBe(1);
  });

  it("should not go below step 0", () => {
    useOnboardingStore.getState().prevStep();
    expect(useOnboardingStore.getState().currentStep).toBe(0);
  });

  it("should have 6 total steps", () => {
    expect(useOnboardingStore.getState().totalSteps).toBe(6);
  });

  it("should not exceed max step on nextStep", () => {
    for (let i = 0; i < 10; i++) {
      useOnboardingStore.getState().nextStep();
    }
    expect(useOnboardingStore.getState().currentStep).toBe(5);
  });

  it("should mark completed via completeOnboarding", () => {
    useOnboardingStore.getState().completeOnboarding();
    expect(useOnboardingStore.getState().completed).toBe(true);
  });

  it("should track library import source selection", () => {
    useOnboardingStore.getState().setImportSource("rekordbox");
    expect(useOnboardingStore.getState().importSource).toBe("rekordbox");
  });

  it("should track history import opt-in", () => {
    expect(useOnboardingStore.getState().historyOptIn).toBe(false);
    useOnboardingStore.getState().setHistoryOptIn(true);
    expect(useOnboardingStore.getState().historyOptIn).toBe(true);
  });

  it("should track community opt-in", () => {
    expect(useOnboardingStore.getState().communityOptIn).toBe(false);
    useOnboardingStore.getState().setCommunityOptIn(true);
    expect(useOnboardingStore.getState().communityOptIn).toBe(true);
  });

  it("should go to a specific step", () => {
    useOnboardingStore.getState().goToStep(3);
    expect(useOnboardingStore.getState().currentStep).toBe(3);
  });

  it("should clamp goToStep within bounds", () => {
    useOnboardingStore.getState().goToStep(-1);
    expect(useOnboardingStore.getState().currentStep).toBe(0);
    useOnboardingStore.getState().goToStep(100);
    expect(useOnboardingStore.getState().currentStep).toBe(5);
  });

  it("should reset all state", () => {
    useOnboardingStore.getState().nextStep();
    useOnboardingStore.getState().setImportSource("serato");
    useOnboardingStore.getState().setCommunityOptIn(true);
    useOnboardingStore.getState().completeOnboarding();
    useOnboardingStore.getState().reset();

    const state = useOnboardingStore.getState();
    expect(state.currentStep).toBe(0);
    expect(state.completed).toBe(false);
    expect(state.importSource).toBeNull();
    expect(state.communityOptIn).toBe(false);
  });
});
