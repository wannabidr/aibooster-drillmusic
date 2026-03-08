import type { AggregateStats, SessionAnalytics, AnalyticsTimeRange } from "@domain/entities/SessionAnalytics";

export interface AnalyticsProvider {
  getSessions(range: AnalyticsTimeRange): Promise<SessionAnalytics[]>;
  getAggregateStats(range: AnalyticsTimeRange): Promise<AggregateStats>;
}
