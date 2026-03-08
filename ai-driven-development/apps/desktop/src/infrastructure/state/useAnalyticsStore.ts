import { create } from "zustand";
import type { AnalyticsDashboardDTO } from "@application/dto/AnalyticsDashboardDTO";
import type { AnalyticsTimeRange } from "@domain/entities/SessionAnalytics";

interface AnalyticsState {
  dashboard: AnalyticsDashboardDTO | null;
  timeRange: AnalyticsTimeRange;
  isLoading: boolean;
  error: string | null;

  setDashboard: (dashboard: AnalyticsDashboardDTO) => void;
  setTimeRange: (range: AnalyticsTimeRange) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clear: () => void;
}

export const useAnalyticsStore = create<AnalyticsState>((set) => ({
  dashboard: null,
  timeRange: "30d",
  isLoading: false,
  error: null,

  setDashboard: (dashboard) => set({ dashboard, error: null }),
  setTimeRange: (timeRange) => set({ timeRange }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
  clear: () => set({ dashboard: null, error: null }),
}));
