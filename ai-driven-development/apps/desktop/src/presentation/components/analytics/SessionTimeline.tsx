import type { SessionTimelineDTO } from "@application/dto/AnalyticsDashboardDTO";

interface SessionTimelineProps {
  data: SessionTimelineDTO | null;
}

const WIDTH = 400;
const HEIGHT = 140;
const PADDING = 24;
const DOT_RADIUS = 6;

export function SessionTimeline({ data }: SessionTimelineProps) {
  if (!data || data.sessions.length === 0) {
    return (
      <div style={styles.empty}>
        No session history available
      </div>
    );
  }

  const sessions = data.sessions;
  const plotW = WIDTH - PADDING * 2;
  const plotH = HEIGHT - PADDING * 2;
  const step = sessions.length > 1 ? plotW / (sessions.length - 1) : plotW / 2;

  const maxTracks = Math.max(1, ...sessions.map((s) => s.trackCount));

  // Energy curve across sessions
  const energyPath = sessions
    .map((s, i) => {
      const x = PADDING + i * step;
      const y = PADDING + plotH - (s.avgEnergy / 100) * plotH;
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Session History</h3>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{ width: "100%", height: "auto" }}
        preserveAspectRatio="xMidYMid meet"
        data-testid="session-timeline-chart"
      >
        {/* Grid */}
        {[0, 50, 100].map((val) => {
          const y = PADDING + plotH - (val / 100) * plotH;
          return (
            <line
              key={val}
              x1={PADDING}
              y1={y}
              x2={WIDTH - PADDING}
              y2={y}
              stroke="var(--border-subtle)"
              strokeWidth={0.5}
              strokeDasharray="2,2"
            />
          );
        })}

        {/* Track count bars (background) */}
        {sessions.map((s, i) => {
          const x = PADDING + i * step;
          const barH = (s.trackCount / maxTracks) * plotH;
          const barW = Math.max(4, step * 0.4);
          return (
            <rect
              key={`bar-${s.sessionId}`}
              data-testid="session-bar"
              x={x - barW / 2}
              y={PADDING + plotH - barH}
              width={barW}
              height={barH}
              rx={2}
              fill="var(--bg-overlay)"
              opacity={0.6}
            />
          );
        })}

        {/* Energy trend line */}
        <path
          data-testid="session-energy-line"
          d={energyPath}
          fill="none"
          stroke="var(--accent-primary)"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Session dots */}
        {sessions.map((s, i) => {
          const x = PADDING + i * step;
          const y = PADDING + plotH - (s.avgEnergy / 100) * plotH;
          const qualityColor =
            s.transitionQuality >= 75
              ? "var(--camelot-compatible)"
              : s.transitionQuality >= 50
                ? "var(--camelot-adjacent)"
                : "var(--camelot-clash)";
          return (
            <circle
              key={s.sessionId}
              data-testid="session-dot"
              cx={x}
              cy={y}
              r={DOT_RADIUS}
              fill={qualityColor}
              stroke="var(--bg-surface)"
              strokeWidth={2}
            />
          );
        })}
      </svg>

      {/* Session list */}
      <div style={styles.sessionList} data-testid="session-list">
        {sessions.map((s) => {
          const date = new Date(s.timestamp);
          const dateStr = date.toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
          });
          return (
            <div key={s.sessionId} style={styles.sessionItem}>
              <span style={styles.sessionDate}>{dateStr}</span>
              <span style={styles.sessionGenre}>{s.topGenre}</span>
              <span style={styles.sessionTracks}>{s.trackCount} tracks</span>
              <span style={styles.sessionQuality}>
                {Math.round(s.transitionQuality)}%
              </span>
            </div>
          );
        })}
      </div>
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
    marginBottom: "var(--space-md)",
  },
  sessionList: {
    display: "flex",
    flexDirection: "column" as const,
    gap: 2,
    marginTop: "var(--space-md)",
    maxHeight: 200,
    overflowY: "auto" as const,
  },
  sessionItem: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-sm)",
    padding: "var(--space-xs) var(--space-sm)",
    borderRadius: "var(--radius-sm)",
    fontSize: "var(--font-size-sm)",
  },
  sessionDate: {
    color: "var(--text-muted)",
    fontFamily: "var(--font-mono)",
    fontSize: "var(--font-size-xs)",
    width: 50,
    flexShrink: 0,
  },
  sessionGenre: {
    color: "var(--text-secondary)",
    flex: 1,
  },
  sessionTracks: {
    color: "var(--text-muted)",
    fontSize: "var(--font-size-xs)",
    fontFamily: "var(--font-mono)",
  },
  sessionQuality: {
    color: "var(--accent-secondary)",
    fontWeight: 600,
    fontFamily: "var(--font-mono)",
    fontSize: "var(--font-size-xs)",
    width: 36,
    textAlign: "right" as const,
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
