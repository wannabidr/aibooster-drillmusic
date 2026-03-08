import { useEffect } from "react";
import type { AnalyticsTimeRange } from "@domain/entities/SessionAnalytics";
import { useAnalytics } from "../hooks/useAnalytics";
import { EnergyReport } from "../components/analytics/EnergyReport";
import { GenreDistribution } from "../components/analytics/GenreDistribution";
import { MixingPatterns } from "../components/analytics/MixingPatterns";
import { SessionTimeline } from "../components/analytics/SessionTimeline";

interface AnalyticsDashboardProps {
  onBack: () => void;
}

const TIME_RANGES: { value: AnalyticsTimeRange; label: string }[] = [
  { value: "7d", label: "7 Days" },
  { value: "30d", label: "30 Days" },
  { value: "90d", label: "90 Days" },
  { value: "all", label: "All Time" },
];

export function AnalyticsDashboard({ onBack }: AnalyticsDashboardProps) {
  const { dashboard, timeRange, isLoading, error, fetchAnalytics } = useAnalytics();

  useEffect(() => {
    fetchAnalytics();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTimeRangeChange = (range: AnalyticsTimeRange) => {
    fetchAnalytics(range);
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <button onClick={onBack} style={styles.backButton} data-testid="back-button">
          ← Back
        </button>
        <h2 style={styles.heading}>Analytics</h2>
        <div style={styles.timeRangeGroup}>
          {TIME_RANGES.map((r) => (
            <button
              key={r.value}
              data-testid={`range-${r.value}`}
              onClick={() => handleTimeRangeChange(r.value)}
              style={{
                ...styles.rangeButton,
                ...(timeRange === r.value ? styles.rangeButtonActive : {}),
              }}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      {isLoading && (
        <div style={styles.loading} data-testid="loading">Loading analytics...</div>
      )}

      {error && (
        <div style={styles.error} data-testid="error">{error}</div>
      )}

      {!isLoading && !error && dashboard && (
        <div style={styles.grid}>
          <div style={styles.fullWidth}>
            <EnergyReport data={dashboard.energy} />
          </div>
          <div style={styles.halfWidth}>
            <GenreDistribution data={dashboard.genres} />
          </div>
          <div style={styles.halfWidth}>
            <MixingPatterns data={dashboard.mixing} />
          </div>
          <div style={styles.fullWidth}>
            <SessionTimeline data={dashboard.timeline} />
          </div>
        </div>
      )}

      {!isLoading && !error && !dashboard && (
        <div style={styles.empty} data-testid="empty-state">
          No analytics data yet. Start mixing to see your insights.
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "var(--space-md)",
    height: "100%",
    overflowY: "auto",
    backgroundColor: "var(--bg-base)",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-md)",
    marginBottom: "var(--space-lg)",
  },
  backButton: {
    background: "none",
    border: "1px solid var(--border-default)",
    borderRadius: "var(--radius-md)",
    color: "var(--text-secondary)",
    padding: "var(--space-xs) var(--space-sm)",
    cursor: "pointer",
    fontSize: "var(--font-size-sm)",
  },
  heading: {
    fontSize: "var(--font-size-xl)",
    fontWeight: 700,
    color: "var(--text-primary)",
    margin: 0,
    flex: 1,
  },
  timeRangeGroup: {
    display: "flex",
    gap: 2,
    backgroundColor: "var(--bg-elevated)",
    borderRadius: "var(--radius-md)",
    padding: 2,
  },
  rangeButton: {
    background: "none",
    border: "none",
    borderRadius: "var(--radius-sm)",
    color: "var(--text-muted)",
    padding: "var(--space-xs) var(--space-sm)",
    cursor: "pointer",
    fontSize: "var(--font-size-xs)",
    fontWeight: 500,
    transition: "var(--transition-fast)",
  },
  rangeButtonActive: {
    backgroundColor: "var(--accent-primary)",
    color: "#fff",
    fontWeight: 600,
  },
  grid: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: "var(--space-md)",
  },
  fullWidth: {
    width: "100%",
  },
  halfWidth: {
    flex: "1 1 calc(50% - var(--space-sm))",
    minWidth: 300,
  },
  loading: {
    textAlign: "center" as const,
    color: "var(--text-muted)",
    padding: "var(--space-2xl)",
    fontSize: "var(--font-size-md)",
  },
  error: {
    textAlign: "center" as const,
    color: "var(--status-error)",
    padding: "var(--space-2xl)",
    fontSize: "var(--font-size-md)",
  },
  empty: {
    textAlign: "center" as const,
    color: "var(--text-muted)",
    padding: "var(--space-2xl)",
    fontSize: "var(--font-size-md)",
  },
};
