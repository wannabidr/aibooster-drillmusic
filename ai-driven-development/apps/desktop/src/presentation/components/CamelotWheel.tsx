import { useMemo } from "react";

interface CamelotWheelProps {
  currentKey?: string;
  selectedKey?: string;
  onKeySelect: (key: string) => void;
}

type PositionState = "current" | "compatible" | "selected" | "default";

const POSITIONS: string[] = [];
for (let n = 1; n <= 12; n++) {
  POSITIONS.push(`${n}A`, `${n}B`);
}

function parseKey(key: string): { num: number; letter: string } | null {
  const match = key.match(/^(\d{1,2})([AB])$/);
  if (!match) return null;
  return { num: parseInt(match[1]!, 10), letter: match[2]! };
}

function circularDistance(a: number, b: number): number {
  const diff = Math.abs(a - b);
  return Math.min(diff, 12 - diff);
}

function isCompatible(current: string, position: string): boolean {
  const c = parseKey(current);
  const p = parseKey(position);
  if (!c || !p) return false;
  const dist = circularDistance(c.num, p.num);
  if (dist === 0 && c.letter !== p.letter) return true; // mode switch
  if (dist === 1 && c.letter === p.letter) return true; // adjacent same letter
  return false;
}

function getPositionState(
  position: string,
  currentKey?: string,
  selectedKey?: string,
): PositionState {
  if (selectedKey && position === selectedKey) return "selected";
  if (currentKey && position === currentKey) return "current";
  if (currentKey && isCompatible(currentKey, position)) return "compatible";
  return "default";
}

const STATE_COLORS: Record<PositionState, string> = {
  current: "var(--accent-primary, #6366f1)",
  compatible: "var(--camelot-compatible, #22c55e)",
  selected: "var(--accent-secondary, #f59e0b)",
  default: "var(--bg-overlay, #374151)",
};

const STATE_TEXT_COLORS: Record<PositionState, string> = {
  current: "#fff",
  compatible: "#fff",
  selected: "#000",
  default: "var(--text-secondary, #9ca3af)",
};

export function CamelotWheel({
  currentKey,
  selectedKey,
  onKeySelect,
}: CamelotWheelProps) {
  const positions = useMemo(() => {
    return POSITIONS.map((pos) => ({
      key: pos,
      state: getPositionState(pos, currentKey, selectedKey),
    }));
  }, [currentKey, selectedKey]);

  const cx = 150;
  const cy = 150;
  const outerRadius = 130;
  const innerRadius = 85;

  return (
    <svg
      viewBox="0 0 300 300"
      style={{ width: "100%", maxWidth: 300, height: "auto" }}
    >
      {positions.map(({ key: pos, state }) => {
        const parsed = parseKey(pos);
        if (!parsed) return null;
        const isB = parsed.letter === "B";
        const radius = isB ? innerRadius : outerRadius;
        // Position around circle: each number gets 30 degrees, A is outer, B is inner
        const angle = ((parsed.num - 1) * 30 - 90) * (Math.PI / 180);
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        const cellRadius = 16;

        return (
          <g
            key={pos}
            data-position={pos}
            data-state={state}
            onClick={() => onKeySelect(pos)}
            style={{ cursor: "pointer" }}
          >
            <circle
              cx={x}
              cy={y}
              r={cellRadius}
              fill={STATE_COLORS[state]}
              stroke={state === "current" ? "#fff" : "transparent"}
              strokeWidth={state === "current" ? 2 : 0}
            />
            <text
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fill={STATE_TEXT_COLORS[state]}
              fontSize={10}
              fontWeight={state === "current" ? 700 : 500}
              fontFamily="var(--font-mono, monospace)"
              style={{ pointerEvents: "none" }}
            >
              {pos}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
