import type { EnergyReportDTO } from "@application/dto/AnalyticsDashboardDTO";

interface EnergyReportProps {
  data: EnergyReportDTO | null;
}

const WIDTH = 400;
const HEIGHT = 160;
const PADDING = 24;

function curveToPath(curve: number[]): string {
  if (curve.length === 0) return "";
  const plotW = WIDTH - PADDING * 2;
  const plotH = HEIGHT - PADDING * 2;
  const step = plotW / Math.max(1, curve.length - 1);

  return curve
    .map((val, i) => {
      const x = PADDING + i * step;
      const y = PADDING + plotH - (val / 100) * plotH;
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");
}

function curveToAreaPath(curve: number[]): string {
  if (curve.length === 0) return "";
  const plotW = WIDTH - PADDING * 2;
  const plotH = HEIGHT - PADDING * 2;
  const step = plotW / Math.max(1, curve.length - 1);
  const baseline = PADDING + plotH;

  const linePart = curve
    .map((val, i) => {
      const x = PADDING + i * step;
      const y = PADDING + plotH - (val / 100) * plotH;
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");

  const lastX = PADDING + (curve.length - 1) * step;
  return `${linePart} L${lastX},${baseline} L${PADDING},${baseline} Z`;
}

export function EnergyReport({ data }: EnergyReportProps) {
  if (!data || data.energyCurve.length === 0) {
    return (
      <div style={styles.empty}>
        No energy data available
      </div>
    );
  }

  const plotW = WIDTH - PADDING * 2;
  const plotH = HEIGHT - PADDING * 2;
  const step = plotW / Math.max(1, data.energyCurve.length - 1);

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Energy Report</h3>
      <div style={styles.statsRow}>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Average</span>
          <span style={styles.statValue}>{Math.round(data.avgEnergy)}</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Peaks</span>
          <span style={styles.statValue}>{data.peaks.length}</span>
        </div>
        <div style={styles.stat}>
          <span style={styles.statLabel}>Valleys</span>
          <span style={styles.statValue}>{data.valleys.length}</span>
        </div>
      </div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{ width: "100%", height: "auto" }}
        preserveAspectRatio="xMidYMid meet"
        data-testid="energy-report-chart"
      >
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((val) => {
          const y = PADDING + plotH - (val / 100) * plotH;
          return (
            <g key={val}>
              <line
                x1={PADDING}
                y1={y}
                x2={WIDTH - PADDING}
                y2={y}
                stroke="var(--border-subtle)"
                strokeWidth={0.5}
                strokeDasharray={val === 0 || val === 100 ? undefined : "2,2"}
              />
              <text
                x={PADDING - 4}
                y={y}
                textAnchor="end"
                dominantBaseline="central"
                fontSize={9}
                fill="var(--text-muted)"
                fontFamily="var(--font-mono)"
              >
                {val}
              </text>
            </g>
          );
        })}

        {/* Area fill */}
        <path
          d={curveToAreaPath(data.energyCurve)}
          fill="var(--accent-primary)"
          opacity={0.15}
        />

        {/* Main curve */}
        <path
          data-testid="energy-curve"
          d={curveToPath(data.energyCurve)}
          fill="none"
          stroke="var(--accent-primary)"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Average line */}
        {(() => {
          const avgY = PADDING + plotH - (data.avgEnergy / 100) * plotH;
          return (
            <line
              data-testid="avg-line"
              x1={PADDING}
              y1={avgY}
              x2={WIDTH - PADDING}
              y2={avgY}
              stroke="var(--accent-energy)"
              strokeWidth={1}
              strokeDasharray="4,4"
              opacity={0.7}
            />
          );
        })()}

        {/* Peak markers */}
        {data.peaks.map((p) => {
          const x = PADDING + p.index * step;
          const y = PADDING + plotH - (p.value / 100) * plotH;
          return (
            <circle
              key={`peak-${p.index}`}
              data-testid="peak-marker"
              cx={x}
              cy={y}
              r={4}
              fill="var(--energy-peak)"
              stroke="var(--bg-base)"
              strokeWidth={1.5}
            />
          );
        })}

        {/* Valley markers */}
        {data.valleys.map((v) => {
          const x = PADDING + v.index * step;
          const y = PADDING + plotH - (v.value / 100) * plotH;
          return (
            <circle
              key={`valley-${v.index}`}
              data-testid="valley-marker"
              cx={x}
              cy={y}
              r={4}
              fill="var(--energy-low)"
              stroke="var(--bg-base)"
              strokeWidth={1.5}
            />
          );
        })}
      </svg>
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
    color: "var(--accent-energy)",
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
