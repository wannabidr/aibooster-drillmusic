import { describe, it, expect, vi } from "vitest";
import { GetAnalytics } from "@application/use-cases/GetAnalytics";
import type { AnalyticsProvider } from "@application/ports/AnalyticsProvider";
import type { SessionAnalytics, AggregateStats } from "@domain/entities/SessionAnalytics";

function createMockProvider(
  sessions: SessionAnalytics[],
  stats: AggregateStats,
): AnalyticsProvider {
  return {
    getSessions: vi.fn().mockResolvedValue(sessions),
    getAggregateStats: vi.fn().mockResolvedValue(stats),
  };
}

const mockSessions: SessionAnalytics[] = [
  {
    sessionId: "s1",
    timestamp: new Date("2026-03-01T22:00:00Z"),
    tracksPlayed: [
      { trackId: "t1", playedAt: new Date(), energy: 60, genre: "Techno", bpm: 128, key: "8A" },
      { trackId: "t2", playedAt: new Date(), energy: 70, genre: "Techno", bpm: 130, key: "7A" },
    ],
    energyCurve: [60, 70],
    genreDistribution: { Techno: 2 },
    bpmRange: [128, 130],
    keyTransitions: [["8A", "7A"]],
    avgTransitionQuality: 85,
  },
  {
    sessionId: "s2",
    timestamp: new Date("2026-03-04T21:00:00Z"),
    tracksPlayed: [
      { trackId: "t3", playedAt: new Date(), energy: 50, genre: "House", bpm: 124, key: "5B" },
    ],
    energyCurve: [50],
    genreDistribution: { House: 1 },
    bpmRange: [124, 124],
    keyTransitions: [],
    avgTransitionQuality: 0,
  },
];

const mockStats: AggregateStats = {
  totalSessions: 2,
  totalTracksPlayed: 3,
  avgSessionLength: 1.5,
  topGenres: [
    { genre: "Techno", count: 2 },
    { genre: "House", count: 1 },
  ],
  bpmDistribution: [
    { range: "120-125", count: 1 },
    { range: "126-130", count: 2 },
  ],
  keyUsage: [
    { key: "8A", count: 1 },
    { key: "7A", count: 1 },
    { key: "5B", count: 1 },
  ],
  energyTrend: [40, 55, 70, 60, 50, 65, 80],
  commonTransitions: [{ from: "8A", to: "7A", count: 5 }],
  avgTransitionQuality: 75,
};

describe("GetAnalytics", () => {
  it("should return a complete dashboard DTO", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.energy).toBeDefined();
    expect(result.genres).toBeDefined();
    expect(result.mixing).toBeDefined();
    expect(result.timeline).toBeDefined();
  });

  it("should detect peaks and valleys in energy curve", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    // energyTrend: [40, 55, 70, 60, 50, 65, 80]
    // peaks at index 2 (70) and 6 (80 — but last element, not counted)
    // valleys at index 4 (50)
    expect(result.energy.peaks.length).toBeGreaterThanOrEqual(1);
    expect(result.energy.valleys.length).toBeGreaterThanOrEqual(1);
  });

  it("should compute average energy", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    const expectedAvg = [40, 55, 70, 60, 50, 65, 80].reduce((a, b) => a + b, 0) / 7;
    expect(result.energy.avgEnergy).toBeCloseTo(expectedAvg, 0);
  });

  it("should compute genre distribution with percentages", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.genres.total).toBe(3);
    expect(result.genres.items.length).toBe(2);
    const techno = result.genres.items.find((g) => g.genre === "Techno");
    expect(techno?.percentage).toBe(67); // 2/3 = 66.7 -> rounded to 67
  });

  it("should extract BPM range from distribution", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.mixing.preferredBpmRange.min).toBe(120);
    expect(result.mixing.preferredBpmRange.max).toBe(126);
  });

  it("should include common transitions", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.mixing.commonTransitions.length).toBe(1);
    expect(result.mixing.commonTransitions[0]?.from).toBe("8A");
    expect(result.mixing.commonTransitions[0]?.to).toBe("7A");
  });

  it("should build session timeline with correct track counts", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.timeline.sessions.length).toBe(2);
    expect(result.timeline.sessions[0]?.trackCount).toBe(2);
    expect(result.timeline.sessions[1]?.trackCount).toBe(1);
  });

  it("should determine top genre per session", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.timeline.sessions[0]?.topGenre).toBe("Techno");
    expect(result.timeline.sessions[1]?.topGenre).toBe("House");
  });

  it("should call provider with correct range", async () => {
    const provider = createMockProvider(mockSessions, mockStats);
    const useCase = new GetAnalytics(provider);
    await useCase.execute("7d");

    expect(provider.getSessions).toHaveBeenCalledWith("7d");
    expect(provider.getAggregateStats).toHaveBeenCalledWith("7d");
  });

  it("should handle empty sessions gracefully", async () => {
    const emptyStats = { ...mockStats, energyTrend: [], topGenres: [], commonTransitions: [], keyUsage: [], bpmDistribution: [] };
    const provider = createMockProvider([], emptyStats);
    const useCase = new GetAnalytics(provider);
    const result = await useCase.execute("30d");

    expect(result.energy.energyCurve).toEqual([]);
    expect(result.energy.avgEnergy).toBe(0);
    expect(result.genres.total).toBe(0);
    expect(result.timeline.sessions).toEqual([]);
  });
});
