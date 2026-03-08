import type { Trajectory } from "@domain/value-objects/EnergyProfile";

interface EnergyOverlay {
  trackId: string;
  curve: number[];
  label: string;
}

interface EnergyGraphProps {
  currentEnergyCurve?: number[];
  overlays?: EnergyOverlay[];
  direction?: Trajectory;
}

const WIDTH = 300;
const HEIGHT = 120;
const PADDING = 16;

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

const DIRECTION_LABELS: Record<Trajectory, string> = {
  build: "Build",
  maintain: "Maintain",
  drop: "Drop",
};

const DIRECTION_COLORS: Record<Trajectory, string> = {
  build: "var(--camelot-compatible, #22c55e)",
  maintain: "var(--text-muted, #9ca3af)",
  drop: "var(--camelot-clash, #ef4444)",
};

export function EnergyGraph({
  currentEnergyCurve,
  overlays,
  direction,
}: EnergyGraphProps) {
  if (!currentEnergyCurve || currentEnergyCurve.length === 0) {
    return (
      <div
        style={{
          textAlign: "center",
          color: "var(--text-muted)",
          padding: "var(--space-md)",
          fontSize: "var(--font-size-sm)",
        }}
      >
        No energy data
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{ width: "100%", height: "auto" }}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((val) => {
          const y = PADDING + (HEIGHT - PADDING * 2) - (val / 100) * (HEIGHT - PADDING * 2);
          return (
            <line
              key={val}
              x1={PADDING}
              y1={y}
              x2={WIDTH - PADDING}
              y2={y}
              stroke="var(--border-subtle, #374151)"
              strokeWidth={0.5}
              strokeDasharray={val === 0 || val === 100 ? undefined : "2,2"}
            />
          );
        })}

        {/* Overlay paths */}
        {overlays?.map((overlay) => (
          <path
            key={overlay.trackId}
            data-role="overlay"
            d={curveToPath(overlay.curve)}
            fill="none"
            stroke="var(--accent-secondary, #f59e0b)"
            strokeWidth={1.5}
            strokeDasharray="4,3"
            opacity={0.6}
          />
        ))}

        {/* Current track path */}
        <path
          data-role="current"
          d={curveToPath(currentEnergyCurve)}
          fill="none"
          stroke="var(--accent-primary, #6366f1)"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {/* Direction indicator */}
      {direction && (
        <div
          style={{
            position: "absolute",
            top: 4,
            right: 8,
            fontSize: "var(--font-size-xs, 11px)",
            fontWeight: 600,
            color: DIRECTION_COLORS[direction],
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <span>
            {direction === "build" ? "\u2197" : direction === "drop" ? "\u2198" : "\u2192"}
          </span>
          <span>{DIRECTION_LABELS[direction]}</span>
        </div>
      )}
    </div>
  );
}
