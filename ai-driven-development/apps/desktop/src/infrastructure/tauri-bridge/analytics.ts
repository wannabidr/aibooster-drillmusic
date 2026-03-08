import type { AnalyticsProvider } from "@application/ports/AnalyticsProvider";
import type { AggregateStats, AnalyticsTimeRange, SessionAnalytics } from "@domain/entities/SessionAnalytics";

function rangeToDays(range: AnalyticsTimeRange): number {
  switch (range) {
    case "7d": return 7;
    case "30d": return 30;
    case "90d": return 90;
    case "all": return 0;
  }
}

export class TauriAnalyticsProvider implements AnalyticsProvider {
  async getSessions(range: AnalyticsTimeRange): Promise<SessionAnalytics[]> {
    const { invoke } = await import("./invoke");
    const days = rangeToDays(range);
    const raw = await invoke<SessionAnalytics[]>("get_analytics_sessions", { days });
    return raw.map((s) => ({
      ...s,
      timestamp: new Date(s.timestamp),
      tracksPlayed: s.tracksPlayed.map((t) => ({
        ...t,
        playedAt: new Date(t.playedAt),
      })),
    }));
  }

  async getAggregateStats(range: AnalyticsTimeRange): Promise<AggregateStats> {
    const { invoke } = await import("./invoke");
    const days = rangeToDays(range);
    return invoke<AggregateStats>("get_analytics_aggregate", { days });
  }
}
