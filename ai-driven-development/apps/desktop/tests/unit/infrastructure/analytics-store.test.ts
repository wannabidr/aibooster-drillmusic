import { describe, it, expect, beforeEach } from "vitest";
import { useAnalyticsStore } from "@infrastructure/state/useAnalyticsStore";
import type { AnalyticsDashboardDTO } from "@application/dto/AnalyticsDashboardDTO";

const mockDashboard: AnalyticsDashboardDTO = {
  energy: {
    energyCurve: [50, 60, 70],
    peaks: [{ index: 2, value: 70 }],
    valleys: [],
    avgEnergy: 60,
  },
  genres: {
    items: [{ genre: "Techno", count: 10, percentage: 100 }],
    total: 10,
  },
  mixing: {
    commonTransitions: [],
    preferredBpmRange: { min: 125, max: 130, avg: 128 },
    keyPreferences: [],
    avgTransitionQuality: 80,
  },
  timeline: { sessions: [] },
};

describe("useAnalyticsStore", () => {
  beforeEach(() => {
    useAnalyticsStore.setState({
      dashboard: null,
      timeRange: "30d",
      isLoading: false,
      error: null,
    });
  });

  it("should have correct initial state", () => {
    const state = useAnalyticsStore.getState();
    expect(state.dashboard).toBeNull();
    expect(state.timeRange).toBe("30d");
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
  });

  it("should set dashboard data", () => {
    useAnalyticsStore.getState().setDashboard(mockDashboard);
    expect(useAnalyticsStore.getState().dashboard).toEqual(mockDashboard);
    expect(useAnalyticsStore.getState().error).toBeNull();
  });

  it("should set time range", () => {
    useAnalyticsStore.getState().setTimeRange("7d");
    expect(useAnalyticsStore.getState().timeRange).toBe("7d");
  });

  it("should set loading state", () => {
    useAnalyticsStore.getState().setLoading(true);
    expect(useAnalyticsStore.getState().isLoading).toBe(true);
  });

  it("should set error and stop loading", () => {
    useAnalyticsStore.getState().setLoading(true);
    useAnalyticsStore.getState().setError("Network error");
    expect(useAnalyticsStore.getState().error).toBe("Network error");
    expect(useAnalyticsStore.getState().isLoading).toBe(false);
  });

  it("should clear dashboard and error", () => {
    useAnalyticsStore.getState().setDashboard(mockDashboard);
    useAnalyticsStore.getState().setError("some error");
    useAnalyticsStore.getState().clear();
    expect(useAnalyticsStore.getState().dashboard).toBeNull();
    expect(useAnalyticsStore.getState().error).toBeNull();
  });

  it("should clear error when setting new dashboard", () => {
    useAnalyticsStore.getState().setError("old error");
    useAnalyticsStore.getState().setDashboard(mockDashboard);
    expect(useAnalyticsStore.getState().error).toBeNull();
  });
});
