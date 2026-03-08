interface RecommendationItem {
  trackId: string;
  title: string;
  artist: string;
  score: number;
  bpmScore: number;
  keyScore: number;
  energyScore: number;
  genreScore: number;
  historyScore: number;
  camelotPosition?: string;
  bpm?: number;
}

interface RecommendationPanelProps {
  recommendations: RecommendationItem[];
  confidence: number;
  isLoading: boolean;
  onSelect: (trackId: string) => void;
  onPreview: (trackId: string) => void;
  glanceable?: boolean;
  performanceMode?: boolean;
}

export function RecommendationPanel({
  recommendations,
  confidence,
  isLoading,
  onSelect,
  onPreview,
  glanceable = false,
  performanceMode = false,
}: RecommendationPanelProps) {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}>Next Track</h2>
        {!performanceMode && (
          <span style={styles.confidence}>
            {confidence > 0 ? `${confidence}% confidence` : ""}
          </span>
        )}
      </div>

      {isLoading ? (
        <div style={styles.loading}>Analyzing...</div>
      ) : recommendations.length === 0 ? (
        <div style={styles.empty}>Select a track to see recommendations</div>
      ) : (
        <div style={styles.list}>
          {recommendations.map((rec, index) => (
            <div key={rec.trackId} style={styles.card} onClick={() => onSelect(rec.trackId)}>
              <div style={styles.cardHeader}>
                <span style={styles.rank}>#{index + 1}</span>
                {glanceable ? (
                  <div style={styles.glanceableRow}>
                    {rec.bpm != null && (
                      <span
                        data-testid={`glanceable-bpm-${rec.trackId}`}
                        style={styles.glanceableBpm}
                      >
                        {rec.bpm}
                      </span>
                    )}
                    {rec.camelotPosition && (
                      <span
                        data-testid={`glanceable-key-${rec.trackId}`}
                        style={styles.glanceableKey}
                      >
                        {rec.camelotPosition}
                      </span>
                    )}
                  </div>
                ) : (
                  <span style={styles.score}>{rec.score}</span>
                )}
              </div>
              <div style={styles.trackInfo}>
                <div style={styles.trackTitle}>{rec.title}</div>
                {!performanceMode && (
                  <div style={styles.trackArtist}>{rec.artist}</div>
                )}
              </div>
              {!performanceMode && (
                <div style={styles.scores}>
                  <ScoreBadge label="BPM" value={rec.bpmScore} detail={rec.bpm?.toString()} />
                  <ScoreBadge label="KEY" value={rec.keyScore} detail={rec.camelotPosition} />
                  <ScoreBadge label="NRG" value={rec.energyScore} />
                  <ScoreBadge label="GNR" value={rec.genreScore} />
                  <ScoreBadge label="HST" value={rec.historyScore} />
                </div>
              )}
              <button
                style={styles.previewBtn}
                onClick={(e) => {
                  e.stopPropagation();
                  onPreview(rec.trackId);
                }}
              >
                Preview
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ScoreBadge({ label, value, detail }: { label: string; value: number; detail?: string }) {
  const color =
    value >= 80 ? "var(--camelot-compatible)" :
    value >= 60 ? "var(--camelot-adjacent)" :
    "var(--camelot-clash)";

  return (
    <div style={{ textAlign: "center", fontSize: "var(--font-size-xs)" }}>
      <div style={{ color: "var(--text-muted)", marginBottom: 2 }}>{label}</div>
      <div style={{ color, fontWeight: 600, fontFamily: "var(--font-mono)" }}>{value}</div>
      {detail && <div style={{ color: "var(--text-muted)", fontSize: 10 }}>{detail}</div>}
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
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "var(--space-md)",
  },
  title: {
    fontSize: "var(--font-size-lg)",
    fontWeight: 600,
  },
  confidence: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-muted)",
  },
  loading: {
    textAlign: "center",
    color: "var(--text-muted)",
    padding: "var(--space-xl)",
  },
  empty: {
    textAlign: "center",
    color: "var(--text-muted)",
    padding: "var(--space-xl)",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "var(--space-sm)",
    overflowY: "auto",
  },
  card: {
    backgroundColor: "var(--bg-elevated)",
    borderRadius: "var(--radius-md)",
    padding: "var(--space-sm) var(--space-md)",
    cursor: "pointer",
    border: "1px solid var(--border-subtle)",
    transition: "border-color var(--transition-fast)",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "var(--space-xs)",
  },
  rank: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-muted)",
  },
  score: {
    fontSize: "var(--font-size-lg)",
    fontWeight: 700,
    color: "var(--accent-primary)",
    fontFamily: "var(--font-mono)",
  },
  glanceableRow: {
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
  trackInfo: {
    marginBottom: "var(--space-sm)",
  },
  trackTitle: {
    fontWeight: 500,
    fontSize: "var(--font-size-sm)",
  },
  trackArtist: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-secondary)",
  },
  scores: {
    display: "flex",
    justifyContent: "space-around",
    marginBottom: "var(--space-sm)",
  },
  previewBtn: {
    width: "100%",
    padding: "var(--space-xs) var(--space-sm)",
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    border: "none",
    borderRadius: "var(--radius-sm)",
    cursor: "pointer",
    fontSize: "var(--font-size-xs)",
    fontWeight: 500,
    transition: "background-color var(--transition-fast)",
  },
};
