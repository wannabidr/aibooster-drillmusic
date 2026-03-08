import { useCallback } from "react";
import { useAnalyticsStore } from "@infrastructure/state/useAnalyticsStore";
import type { AnalyticsProvider } from "@application/ports/AnalyticsProvider";
import type { AnalyticsTimeRange } from "@domain/entities/SessionAnalytics";
import { GetAnalytics } from "@application/use-cases/GetAnalytics";

let cachedProvider: AnalyticsProvider | null = null;

function getProvider(): AnalyticsProvider {
  if (!cachedProvider) {
    // Lazy import to avoid circular deps and allow mocking
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { TauriAnalyticsProvider } = require("@infrastructure/tauri-bridge/analytics");
    cachedProvider = new TauriAnalyticsProvider();
  }
  return cachedProvider;
}

export function setAnalyticsProvider(provider: AnalyticsProvider) {
  cachedProvider = provider;
}

export function useAnalytics() {
  const store = useAnalyticsStore();

  const fetchAnalytics = useCallback(
    async (range?: AnalyticsTimeRange) => {
      const timeRange = range ?? store.timeRange;
      if (range && range !== store.timeRange) {
        store.setTimeRange(range);
      }
      store.setLoading(true);
      try {
        const useCase = new GetAnalytics(getProvider());
        const dashboard = await useCase.execute(timeRange);
        store.setDashboard(dashboard);
      } catch (err) {
        store.setError(err instanceof Error ? err.message : "Failed to load analytics");
      } finally {
        store.setLoading(false);
      }
    },
    [store],
  );

  return {
    dashboard: store.dashboard,
    timeRange: store.timeRange,
    isLoading: store.isLoading,
    error: store.error,
    fetchAnalytics,
  };
}
