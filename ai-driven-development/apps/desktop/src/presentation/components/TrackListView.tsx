interface TrackListItem {
  id: string;
  title: string;
  artist: string;
  bpm?: number;
  key?: string;
  energy?: number;
  durationMs?: number;
}

interface TrackListViewProps {
  tracks: TrackListItem[];
  selectedTrackId?: string;
  onTrackSelect: (trackId: string) => void;
}

export function TrackListView({ tracks, selectedTrackId, onTrackSelect }: TrackListViewProps) {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={{ ...styles.cell, flex: 3 }}>Title</span>
        <span style={{ ...styles.cell, flex: 2 }}>Artist</span>
        <span style={{ ...styles.cell, flex: 0.5, textAlign: "center" }}>BPM</span>
        <span style={{ ...styles.cell, flex: 0.5, textAlign: "center" }}>Key</span>
        <span style={{ ...styles.cell, flex: 0.5, textAlign: "center" }}>Energy</span>
      </div>
      <div style={styles.list}>
        {tracks.map((track) => (
          <div
            key={track.id}
            style={{
              ...styles.row,
              backgroundColor:
                track.id === selectedTrackId ? "var(--bg-active)" : "transparent",
            }}
            onClick={() => onTrackSelect(track.id)}
          >
            <span style={{ ...styles.cell, flex: 3 }}>{track.title}</span>
            <span style={{ ...styles.cell, flex: 2, color: "var(--text-secondary)" }}>
              {track.artist}
            </span>
            <span style={{ ...styles.cell, flex: 0.5, textAlign: "center", fontFamily: "var(--font-mono)", fontSize: "var(--font-size-sm)" }}>
              {track.bpm ?? "-"}
            </span>
            <span style={{ ...styles.cell, flex: 0.5, textAlign: "center", fontFamily: "var(--font-mono)", fontSize: "var(--font-size-sm)" }}>
              {track.key ?? "-"}
            </span>
            <span style={{ ...styles.cell, flex: 0.5, textAlign: "center" }}>
              {track.energy != null ? <EnergyBar value={track.energy} /> : "-"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EnergyBar({ value }: { value: number }) {
  const color =
    value >= 80 ? "var(--energy-high)" :
    value >= 50 ? "var(--energy-mid)" :
    "var(--energy-low)";

  return (
    <div style={{ width: 40, height: 6, backgroundColor: "var(--bg-overlay)", borderRadius: "var(--radius-full)" }}>
      <div style={{ width: `${value}%`, height: "100%", backgroundColor: color, borderRadius: "var(--radius-full)", transition: "width var(--transition-normal)" }} />
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
  },
  header: {
    display: "flex",
    padding: "var(--space-sm) var(--space-md)",
    borderBottom: "1px solid var(--border-subtle)",
    color: "var(--text-muted)",
    fontSize: "var(--font-size-xs)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  list: {
    flex: 1,
    overflowY: "auto",
  },
  row: {
    display: "flex",
    padding: "var(--space-sm) var(--space-md)",
    cursor: "pointer",
    borderBottom: "1px solid var(--border-subtle)",
    transition: "background-color var(--transition-fast)",
  },
  cell: {
    display: "flex",
    alignItems: "center",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    paddingRight: "var(--space-sm)",
  },
};
