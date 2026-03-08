import type { AnalyticsProvider } from "@application/ports/AnalyticsProvider";
import type { AnalyticsDashboardDTO } from "@application/dto/AnalyticsDashboardDTO";
import type { AnalyticsTimeRange, SessionAnalytics, AggregateStats } from "@domain/entities/SessionAnalytics";

export class GetAnalytics {
  constructor(private readonly provider: AnalyticsProvider) {}

  async execute(range: AnalyticsTimeRange): Promise<AnalyticsDashboardDTO> {
    const [sessions, stats] = await Promise.all([
      this.provider.getSessions(range),
      this.provider.getAggregateStats(range),
    ]);

    return {
      energy: this.buildEnergyReport(sessions, stats),
      genres: this.buildGenreDistribution(stats),
      mixing: this.buildMixingPatterns(stats),
      timeline: this.buildSessionTimeline(sessions),
    };
  }

  private buildEnergyReport(
    sessions: SessionAnalytics[],
    stats: AggregateStats,
  ): AnalyticsDashboardDTO["energy"] {
    const curve = stats.energyTrend;
    const peaks: { index: number; value: number }[] = [];
    const valleys: { index: number; value: number }[] = [];

    for (let i = 1; i < curve.length - 1; i++) {
      const prev = curve[i - 1]!;
      const curr = curve[i]!;
      const next = curve[i + 1]!;
      if (curr > prev && curr > next) {
        peaks.push({ index: i, value: curr });
      } else if (curr < prev && curr < next) {
        valleys.push({ index: i, value: curr });
      }
    }

    const avgEnergy =
      curve.length > 0
        ? curve.reduce((sum, v) => sum + v, 0) / curve.length
        : 0;

    return { energyCurve: curve, peaks, valleys, avgEnergy };
  }

  private buildGenreDistribution(
    stats: AggregateStats,
  ): AnalyticsDashboardDTO["genres"] {
    const total = stats.topGenres.reduce((sum, g) => sum + g.count, 0);
    const items = stats.topGenres.map((g) => ({
      genre: g.genre,
      count: g.count,
      percentage: total > 0 ? Math.round((g.count / total) * 100) : 0,
    }));
    return { items, total };
  }

  private buildMixingPatterns(
    stats: AggregateStats,
  ): AnalyticsDashboardDTO["mixing"] {
    const bpms = stats.bpmDistribution;
    let min = Infinity;
    let max = -Infinity;
    let totalBpm = 0;
    let totalCount = 0;

    for (const bucket of bpms) {
      const match = bucket.range.match(/(\d+)/);
      if (match) {
        const val = parseInt(match[1]!, 10);
        if (val < min) min = val;
        if (val > max) max = val;
        totalBpm += val * bucket.count;
        totalCount += bucket.count;
      }
    }

    return {
      commonTransitions: stats.commonTransitions,
      preferredBpmRange: {
        min: min === Infinity ? 0 : min,
        max: max === -Infinity ? 0 : max,
        avg: totalCount > 0 ? Math.round(totalBpm / totalCount) : 0,
      },
      keyPreferences: stats.keyUsage,
      avgTransitionQuality: stats.avgTransitionQuality,
    };
  }

  private buildSessionTimeline(
    sessions: SessionAnalytics[],
  ): AnalyticsDashboardDTO["timeline"] {
    return {
      sessions: sessions.map((s) => {
        const avgEnergy =
          s.energyCurve.length > 0
            ? s.energyCurve.reduce((sum, v) => sum + v, 0) / s.energyCurve.length
            : 0;

        const genres = Object.entries(s.genreDistribution);
        const topGenre =
          genres.length > 0
            ? genres.sort((a, b) => b[1] - a[1])[0]![0]
            : "Unknown";

        return {
          sessionId: s.sessionId,
          timestamp: s.timestamp.toISOString(),
          trackCount: s.tracksPlayed.length,
          avgEnergy: Math.round(avgEnergy),
          topGenre,
          transitionQuality: s.avgTransitionQuality,
        };
      }),
    };
  }
}
