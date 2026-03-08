export interface EnergyReportDTO {
  energyCurve: number[];
  peaks: { index: number; value: number }[];
  valleys: { index: number; value: number }[];
  avgEnergy: number;
}

export interface GenreDistributionDTO {
  items: { genre: string; count: number; percentage: number }[];
  total: number;
}

export interface MixingPatternsDTO {
  commonTransitions: { from: string; to: string; count: number }[];
  preferredBpmRange: { min: number; max: number; avg: number };
  keyPreferences: { key: string; count: number }[];
  avgTransitionQuality: number;
}

export interface SessionTimelineDTO {
  sessions: {
    sessionId: string;
    timestamp: string;
    trackCount: number;
    avgEnergy: number;
    topGenre: string;
    transitionQuality: number;
  }[];
}

export interface AnalyticsDashboardDTO {
  energy: EnergyReportDTO;
  genres: GenreDistributionDTO;
  mixing: MixingPatternsDTO;
  timeline: SessionTimelineDTO;
}
