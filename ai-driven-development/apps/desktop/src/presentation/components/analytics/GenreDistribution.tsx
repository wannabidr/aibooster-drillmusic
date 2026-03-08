import type { GenreDistributionDTO } from "@application/dto/AnalyticsDashboardDTO";

interface GenreDistributionProps {
  data: GenreDistributionDTO | null;
}

const COLORS = [
  "var(--accent-primary)",
  "var(--accent-secondary)",
  "var(--accent-warm)",
  "var(--accent-energy)",
  "var(--status-info)",
  "var(--camelot-compatible)",
  "var(--energy-high)",
  "var(--text-muted)",
];

const SIZE = 200;
const CENTER = SIZE / 2;
const RADIUS = 80;
const INNER_RADIUS = 45;

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function arcPath(
  cx: number,
  cy: number,
  outerR: number,
  innerR: number,
  startAngle: number,
  endAngle: number,
): string {
  const sweep = endAngle - startAngle;
  const largeArc = sweep > 180 ? 1 : 0;
  const outerStart = polarToCartesian(cx, cy, outerR, startAngle);
  const outerEnd = polarToCartesian(cx, cy, outerR, endAngle);
  const innerStart = polarToCartesian(cx, cy, innerR, endAngle);
  const innerEnd = polarToCartesian(cx, cy, innerR, startAngle);

  return [
    `M ${outerStart.x} ${outerStart.y}`,
    `A ${outerR} ${outerR} 0 ${largeArc} 1 ${outerEnd.x} ${outerEnd.y}`,
    `L ${innerStart.x} ${innerStart.y}`,
    `A ${innerR} ${innerR} 0 ${largeArc} 0 ${innerEnd.x} ${innerEnd.y}`,
    "Z",
  ].join(" ");
}

export function GenreDistribution({ data }: GenreDistributionProps) {
  if (!data || data.items.length === 0) {
    return (
      <div style={styles.empty}>
        No genre data available
      </div>
    );
  }

  let currentAngle = 0;
  const slices = data.items.map((item, i) => {
    const sweepAngle = (item.count / data.total) * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sweepAngle;
    currentAngle = endAngle;
    return {
      ...item,
      startAngle,
      endAngle,
      color: COLORS[i % COLORS.length]!,
    };
  });

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Genre Distribution</h3>
      <div style={styles.chartRow}>
        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          style={{ width: SIZE, height: SIZE, flexShrink: 0 }}
          data-testid="genre-chart"
        >
          {slices.map((slice) => (
            <path
              key={slice.genre}
              data-testid="genre-slice"
              d={arcPath(
                CENTER,
                CENTER,
                RADIUS,
                INNER_RADIUS,
                slice.startAngle,
                slice.endAngle,
              )}
              fill={slice.color}
              stroke="var(--bg-surface)"
              strokeWidth={2}
            />
          ))}
          {/* Center text */}
          <text
            x={CENTER}
            y={CENTER - 6}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={18}
            fontWeight={700}
            fontFamily="var(--font-mono)"
            fill="var(--text-primary)"
          >
            {data.total}
          </text>
          <text
            x={CENTER}
            y={CENTER + 12}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={10}
            fill="var(--text-muted)"
          >
            tracks
          </text>
        </svg>
        <div style={styles.legend}>
          {slices.map((slice) => (
            <div key={slice.genre} style={styles.legendItem}>
              <span
                style={{
                  ...styles.legendDot,
                  backgroundColor: slice.color,
                }}
              />
              <span style={styles.legendLabel}>{slice.genre}</span>
              <span style={styles.legendValue}>{slice.percentage}%</span>
            </div>
          ))}
        </div>
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
  chartRow: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-lg)",
  },
  legend: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "var(--space-xs)",
    flex: 1,
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-sm)",
    fontSize: "var(--font-size-sm)",
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: "var(--radius-sm)",
    flexShrink: 0,
  },
  legendLabel: {
    color: "var(--text-secondary)",
    flex: 1,
  },
  legendValue: {
    color: "var(--text-primary)",
    fontFamily: "var(--font-mono)",
    fontWeight: 600,
    fontSize: "var(--font-size-xs)",
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
