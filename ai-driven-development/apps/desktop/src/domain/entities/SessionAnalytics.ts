export interface TrackPlayEvent {
  trackId: string;
  playedAt: Date;
  energy: number;
  genre: string;
  bpm: number;
  key: string;
}

export interface SessionAnalytics {
  sessionId: string;
  timestamp: Date;
  tracksPlayed: TrackPlayEvent[];
  energyCurve: number[];
  genreDistribution: Record<string, number>;
  bpmRange: [number, number];
  keyTransitions: [string, string][];
  avgTransitionQuality: number;
}

export interface AggregateStats {
  totalSessions: number;
  totalTracksPlayed: number;
  avgSessionLength: number;
  topGenres: { genre: string; count: number }[];
  bpmDistribution: { range: string; count: number }[];
  keyUsage: { key: string; count: number }[];
  energyTrend: number[];
  commonTransitions: { from: string; to: string; count: number }[];
  avgTransitionQuality: number;
}

export type AnalyticsTimeRange = "7d" | "30d" | "90d" | "all";
