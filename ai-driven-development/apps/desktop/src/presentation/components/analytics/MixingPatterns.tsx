import type { MixingPatternsDTO } from "@application/dto/AnalyticsDashboardDTO";

interface MixingPatternsProps {
  data: MixingPatternsDTO | null;
}

const BAR_HEIGHT = 24;
const BAR_GAP = 6;
const BAR_WIDTH = 200;
const LABEL_WIDTH = 60;

export function MixingPatterns({ data }: MixingPatternsProps) {
  if (!data) {
    return (
      <div style={styles.empty}>
        No mixing data available
      </div>
    );
  }

  const maxTransitionCount = Math.max(
    1,
    ...data.commonTransitions.map((t) => t.count),
  );
  const maxKeyCount = Math.max(1, ...data.keyPreferences.map((k) => k.count));
  const topTransitions = data.commonTransitions.slice(0, 8);
  const topKeys = data.keyPreferences.slice(0, 12);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Mixing Patterns</h3>

      {/* Stats row */}
      <div style={styles.statsRow}>
        <div style={styles.stat}>
          <span style={styles.statLabel}>BPM Range</span>
          <span style={styles.statValue}>
            {data.preferredBpmRange.min}–{data.preferredBpmRange.max}
          </span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Avg BPM</span>
          <span style={styles.statValue}>{data.preferredBpmRange.avg}</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Transition Quality</span>
          <span style={styles.statValue}>
            {Math.round(data.avgTransitionQuality)}%
          </span>
        </div>
      </div>

      {/* Common transitions */}
      {topTransitions.length > 0 && (
        <div style={styles.section}>
          <h4 style={styles.sectionTitle}>Common Transitions</h4>
          <svg
            viewBox={`0 0 ${LABEL_WIDTH + BAR_WIDTH + 40} ${topTransitions.length * (BAR_HEIGHT + BAR_GAP)}`}
            style={{ width: "100%", height: "auto" }}
            data-testid="transitions-chart"
          >
            {topTransitions.map((t, i) => {
              const y = i * (BAR_HEIGHT + BAR_GAP);
              const barW = (t.count / maxTransitionCount) * BAR_WIDTH;
              return (
                <g key={`${t.from}-${t.to}`} data-testid="transition-bar">
                  <text
                    x={LABEL_WIDTH - 4}
                    y={y + BAR_HEIGHT / 2}
                    textAnchor="end"
                    dominantBaseline="central"
                    fontSize={10}
                    fill="var(--text-secondary)"
                    fontFamily="var(--font-mono)"
                  >
                    {t.from} → {t.to}
                  </text>
                  <rect
                    x={LABEL_WIDTH}
                    y={y + 2}
                    width={barW}
                    height={BAR_HEIGHT - 4}
                    rx={3}
                    fill="var(--accent-secondary)"
                    opacity={0.8}
                  />
                  <text
                    x={LABEL_WIDTH + barW + 6}
                    y={y + BAR_HEIGHT / 2}
                    dominantBaseline="central"
                    fontSize={10}
                    fill="var(--text-muted)"
                    fontFamily="var(--font-mono)"
                  >
                    {t.count}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* Key preferences */}
      {topKeys.length > 0 && (
        <div style={styles.section}>
          <h4 style={styles.sectionTitle}>Key Preferences</h4>
          <svg
            viewBox={`0 0 ${LABEL_WIDTH + BAR_WIDTH + 40} ${topKeys.length * (BAR_HEIGHT + BAR_GAP)}`}
            style={{ width: "100%", height: "auto" }}
            data-testid="keys-chart"
          >
            {topKeys.map((k, i) => {
              const y = i * (BAR_HEIGHT + BAR_GAP);
              const barW = (k.count / maxKeyCount) * BAR_WIDTH;
              return (
                <g key={k.key} data-testid="key-bar">
                  <text
                    x={LABEL_WIDTH - 4}
                    y={y + BAR_HEIGHT / 2}
                    textAnchor="end"
                    dominantBaseline="central"
                    fontSize={10}
                    fill="var(--text-secondary)"
                    fontFamily="var(--font-mono)"
                  >
                    {k.key}
                  </text>
                  <rect
                    x={LABEL_WIDTH}
                    y={y + 2}
                    width={barW}
                    height={BAR_HEIGHT - 4}
                    rx={3}
                    fill="var(--camelot-compatible)"
                    opacity={0.8}
                  />
                  <text
                    x={LABEL_WIDTH + barW + 6}
                    y={y + BAR_HEIGHT / 2}
                    dominantBaseline="central"
                    fontSize={10}
                    fill="var(--text-muted)"
                    fontFamily="var(--font-mono)"
                  >
                    {k.count}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: "var(--bg-surface)",
    borderRadius: "var(--radius-lg)",
    padding: "var(--space-md)",
    border: "1px solid var(--border-subtle)",
  },
  title: {
    fontSize: "var(--font-size-md)",
    fontWeight: 600,
    color: "var(--text-primary)",
    margin: 0,
    marginBottom: "var(--space-sm)",
  },
  statsRow: {
    display: "flex",
    gap: "var(--space-lg)",
    marginBottom: "var(--space-md)",
  },
  stat: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 2,
  },
  statLabel: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-muted)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
  },
  statValue: {
    fontSize: "var(--font-size-lg)",
    fontWeight: 700,
    fontFamily: "var(--font-mono)",
    color: "var(--accent-secondary)",
  },
  section: {
    marginTop: "var(--space-md)",
  },
  sectionTitle: {
    fontSize: "var(--font-size-sm)",
    fontWeight: 600,
    color: "var(--text-secondary)",
    margin: 0,
    marginBottom: "var(--space-sm)",
  },
  empty: {
    textAlign: "center" as const,
    color: "var(--text-muted)",
    padding: "var(--space-xl)",
    fontSize: "var(--font-size-sm)",
    backgroundColor: "var(--bg-surface)",
    borderRadius: "var(--radius-lg)",
    border: "1px solid var(--border-subtle)",
  },
};
