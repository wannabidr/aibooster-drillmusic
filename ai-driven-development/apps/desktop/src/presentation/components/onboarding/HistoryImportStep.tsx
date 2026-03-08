import { useOnboardingStore } from "../../../infrastructure/state/useOnboardingStore";

export function HistoryImportStep() {
  const { historyOptIn, setHistoryOptIn, nextStep } = useOnboardingStore();

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Import Mix History</h2>
      <p style={styles.description}>
        Optionally import your past mix history. This helps the AI learn your mixing patterns and make better recommendations.
      </p>
      <label style={styles.optInRow}>
        <input
          type="checkbox"
          checked={historyOptIn}
          onChange={(e) => setHistoryOptIn(e.target.checked)}
          style={styles.checkbox}
        />
        <span>Import mix history from my DJ software</span>
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
    maxWidth: 480,
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
    lineHeight: 1.6,
  },
  optInRow: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-sm)",
    padding: "var(--space-md)",
    backgroundColor: "var(--bg-surface)",
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
