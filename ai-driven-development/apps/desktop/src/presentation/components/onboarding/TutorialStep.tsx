import { useOnboardingStore } from "../../../infrastructure/state/useOnboardingStore";

const TIPS = [
  {
    title: "Track List",
    description: "Browse your library. Click any track to select it and get instant recommendations.",
  },
  {
    title: "Camelot Wheel",
    description: "See harmonic compatibility at a glance. Click a key to filter recommendations.",
  },
  {
    title: "Recommendations",
    description: "AI-ranked next tracks with BPM, key, energy, and genre scores. Hit Preview to hear the blend.",
  },
  {
    title: "Now Playing",
    description: "Glanceable BPM, key, and energy for the current track. Time remaining warns you when to mix.",
  },
];

export function TutorialStep() {
  const { nextStep } = useOnboardingStore();

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Quick Tour</h2>
      <p style={styles.description}>
        Here's a quick overview of the key features.
      </p>
      <div style={styles.tips}>
        {TIPS.map((tip, i) => (
          <div key={i} style={styles.tip}>
            <div style={styles.tipNumber}>{i + 1}</div>
            <div>
              <h4 style={styles.tipTitle}>{tip.title}</h4>
              <p style={styles.tipDesc}>{tip.description}</p>
            </div>
          </div>
        ))}
      </div>
      <button style={styles.continueBtn} onClick={nextStep}>
        Continue
      </button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    maxWidth: 520,
  },
  title: {
    fontSize: "var(--font-size-xl)",
    fontWeight: 600,
    marginBottom: "var(--space-sm)",
  },
  description: {
    color: "var(--text-secondary)",
    fontSize: "var(--font-size-sm)",
    marginBottom: "var(--space-xl)",
  },
  tips: {
    display: "flex",
    flexDirection: "column",
    gap: "var(--space-md)",
    width: "100%",
    marginBottom: "var(--space-xl)",
  },
  tip: {
    display: "flex",
    gap: "var(--space-md)",
    padding: "var(--space-md)",
    backgroundColor: "var(--bg-surface)",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--border-subtle)",
    textAlign: "left",
  },
  tipNumber: {
    width: 32,
    height: 32,
    borderRadius: "var(--radius-full)",
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: 700,
    fontSize: "var(--font-size-sm)",
    flexShrink: 0,
  },
  tipTitle: {
    fontSize: "var(--font-size-sm)",
    fontWeight: 600,
    marginBottom: "var(--space-xs)",
  },
  tipDesc: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-secondary)",
    lineHeight: 1.5,
  },
  continueBtn: {
    padding: "var(--space-sm) var(--space-xl)",
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    border: "none",
    borderRadius: "var(--radius-md)",
    fontSize: "var(--font-size-md)",
    fontWeight: 600,
    cursor: "pointer",
  },
};
