interface NowPlayingBarProps {
  title?: string;
  artist?: string;
  bpm?: number;
  camelotKey?: string;
  energy?: number;
  elapsedMs?: number;
  durationMs?: number;
  glanceable?: boolean;
}

export function NowPlayingBar({
  title,
  artist,
  bpm,
  camelotKey,
  energy,
  elapsedMs,
  durationMs,
  glanceable = false,
}: NowPlayingBarProps) {
  if (!title) {
    return (
      <div style={styles.container}>
        <span style={styles.empty}>No track playing</span>
      </div>
    );
  }

  const progress = durationMs && elapsedMs ? (elapsedMs / durationMs) * 100 : 0;

  return (
    <div style={styles.container}>
      <div style={styles.trackInfo}>
        <div style={styles.title}>{title}</div>
        <div style={styles.artist}>{artist}</div>
      </div>
      {glanceable ? (
        <div style={styles.glanceableMetadata}>
          {bpm != null && (
            <span data-testid="glanceable-bpm" style={styles.glanceableBpm}>
              {bpm}
            </span>
          )}
          {camelotKey && (
            <span data-testid="glanceable-key" style={styles.glanceableKey}>
              {camelotKey}
            </span>
          )}
          {energy != null && (
            <span style={styles.glanceableEnergy}>E{energy}</span>
          )}
        </div>
      ) : (
        <div style={styles.metadata}>
          {bpm != null && (
            <span style={styles.badge}>{bpm} BPM</span>
          )}
          {camelotKey && (
            <span style={styles.badge}>{camelotKey}</span>
          )}
          {energy != null && (
            <span style={styles.badge}>E: {energy}</span>
          )}
        </div>
      )}
      <div style={styles.progressContainer}>
        <div style={{ ...styles.progressBar, width: `${progress}%` }} />
      </div>
      {durationMs && (
        <div style={styles.time}>
          {formatTime(elapsedMs ?? 0)} / {formatTime(durationMs)}
          <span style={styles.remaining}>
            {" "}-{formatTime(durationMs - (elapsedMs ?? 0))}
          </span>
        </div>
      )}
    </div>
  );
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-md)",
    width: "100%",
    height: "100%",
  },
  empty: {
    color: "var(--text-muted)",
    fontSize: "var(--font-size-sm)",
  },
  trackInfo: {
    minWidth: 200,
  },
  title: {
    fontWeight: 500,
    fontSize: "var(--font-size-sm)",
  },
  artist: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-secondary)",
  },
  metadata: {
    display: "flex",
    gap: "var(--space-sm)",
  },
  badge: {
    padding: "2px var(--space-sm)",
    backgroundColor: "var(--bg-overlay)",
    borderRadius: "var(--radius-sm)",
    fontSize: "var(--font-size-xs)",
    fontFamily: "var(--font-mono)",
    color: "var(--text-secondary)",
  },
  glanceableMetadata: {
    display: "flex",
    gap: "var(--space-md)",
    alignItems: "baseline",
  },
  glanceableBpm: {
    fontSize: "var(--font-size-2xl)",
    fontWeight: 700,
    fontFamily: "var(--font-mono)",
    color: "var(--accent-energy)",
    lineHeight: 1,
  },
  glanceableKey: {
    fontSize: "var(--font-size-xl)",
    fontWeight: 700,
    fontFamily: "var(--font-mono)",
    color: "var(--camelot-compatible)",
    lineHeight: 1,
  },
  glanceableEnergy: {
    fontSize: "var(--font-size-lg)",
    fontWeight: 600,
    fontFamily: "var(--font-mono)",
    color: "var(--accent-secondary)",
    lineHeight: 1,
  },
  progressContainer: {
    flex: 1,
    height: 4,
    backgroundColor: "var(--bg-overlay)",
    borderRadius: "var(--radius-full)",
    overflow: "hidden",
  },
  progressBar: {
    height: "100%",
    backgroundColor: "var(--accent-primary)",
    borderRadius: "var(--radius-full)",
    transition: "width var(--transition-normal)",
  },
  time: {
    fontSize: "var(--font-size-xs)",
    fontFamily: "var(--font-mono)",
    color: "var(--text-muted)",
    whiteSpace: "nowrap",
  },
  remaining: {
    color: "var(--camelot-clash, #ef4444)",
    fontSize: "var(--font-size-xs)",
  },
};
