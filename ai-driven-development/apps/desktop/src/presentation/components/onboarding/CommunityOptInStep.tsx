import { useOnboardingStore } from "../../../infrastructure/state/useOnboardingStore";

export function CommunityOptInStep() {
  const { communityOptIn, setCommunityOptIn, nextStep } = useOnboardingStore();

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Join the Community</h2>
      <p style={styles.description}>
        Help improve recommendations for everyone. When you opt in, we share anonymous,
        aggregated data -- never your personal library or playlists.
      </p>
      <div style={styles.privacyBox}>
        <h4 style={styles.privacyTitle}>What we share:</h4>
        <ul style={styles.privacyList}>
          <li>Anonymous genre and BPM trends (no track titles or artists)</li>
          <li>Aggregated transition success rates</li>
          <li>Anonymized energy flow patterns</li>
        </ul>
        <h4 style={styles.privacyTitle}>What we never share:</h4>
        <ul style={styles.privacyList}>
          <li>Your personal track library</li>
          <li>Your playlists or set lists</li>
          <li>Any personally identifiable information</li>
        </ul>
      </div>
      <label style={styles.optInRow}>
        <input
          type="checkbox"
          checked={communityOptIn}
          onChange={(e) => setCommunityOptIn(e.target.checked)}
          aria-label="I want to join the community and share anonymous data"
          style={styles.checkbox}
        />
        <span>I want to join the community and share anonymous data</span>
      </label>
      <div style={styles.actions}>
        <button style={styles.skipBtn} onClick={nextStep}>
          Skip
        </button>
        <button style={styles.continueBtn} onClick={nextStep}>
          Continue
        </button>
      </div>
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
    marginBottom: "var(--space-lg)",
    lineHeight: 1.6,
  },
  privacyBox: {
    padding: "var(--space-md)",
    backgroundColor: "var(--bg-surface)",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--border-subtle)",
    marginBottom: "var(--space-lg)",
    textAlign: "left",
    width: "100%",
  },
  privacyTitle: {
    fontSize: "var(--font-size-sm)",
    fontWeight: 600,
    marginBottom: "var(--space-xs)",
    marginTop: "var(--space-sm)",
  },
  privacyList: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-secondary)",
    lineHeight: 1.8,
    paddingLeft: "var(--space-lg)",
  },
  optInRow: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-sm)",
    padding: "var(--space-md)",
    backgroundColor: "var(--bg-elevated)",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--border-subtle)",
    marginBottom: "var(--space-xl)",
    cursor: "pointer",
    fontSize: "var(--font-size-sm)",
  },
  checkbox: {
    width: 18,
    height: 18,
    accentColor: "var(--accent-primary)",
  },
  actions: {
    display: "flex",
    gap: "var(--space-md)",
  },
  skipBtn: {
    padding: "var(--space-sm) var(--space-xl)",
    backgroundColor: "transparent",
    color: "var(--text-secondary)",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "var(--border-default)",
    borderRadius: "var(--radius-md)",
    fontSize: "var(--font-size-sm)",
    cursor: "pointer",
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
