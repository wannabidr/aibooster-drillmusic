import { describe, it, expect, beforeEach } from "vitest";
import { useThemeStore } from "@infrastructure/state/useThemeStore";

describe("useThemeStore", () => {
  beforeEach(() => {
    useThemeStore.setState({
      highContrast: false,
      performanceMode: false,
    });
  });

  it("should default to standard mode (no high-contrast, no performance mode)", () => {
    const state = useThemeStore.getState();
    expect(state.highContrast).toBe(false);
    expect(state.performanceMode).toBe(false);
  });

  it("should toggle high-contrast mode", () => {
    useThemeStore.getState().toggleHighContrast();
    expect(useThemeStore.getState().highContrast).toBe(true);
    useThemeStore.getState().toggleHighContrast();
    expect(useThemeStore.getState().highContrast).toBe(false);
  });

  it("should toggle performance mode", () => {
    useThemeStore.getState().togglePerformanceMode();
    expect(useThemeStore.getState().performanceMode).toBe(true);
    useThemeStore.getState().togglePerformanceMode();
    expect(useThemeStore.getState().performanceMode).toBe(false);
  });

  it("should allow setting high-contrast directly", () => {
    useThemeStore.getState().setHighContrast(true);
    expect(useThemeStore.getState().highContrast).toBe(true);
    useThemeStore.getState().setHighContrast(false);
    expect(useThemeStore.getState().highContrast).toBe(false);
  });

  it("should allow setting performance mode directly", () => {
    useThemeStore.getState().setPerformanceMode(true);
    expect(useThemeStore.getState().performanceMode).toBe(true);
    useThemeStore.getState().setPerformanceMode(false);
    expect(useThemeStore.getState().performanceMode).toBe(false);
  });

  it("should support both modes simultaneously", () => {
    useThemeStore.getState().setHighContrast(true);
    useThemeStore.getState().setPerformanceMode(true);
    const state = useThemeStore.getState();
    expect(state.highContrast).toBe(true);
    expect(state.performanceMode).toBe(true);
  });
});
