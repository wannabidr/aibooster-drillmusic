import { useOnboardingStore } from "../../../infrastructure/state/useOnboardingStore";

export function ReadyScreen() {
  const { completeOnboarding } = useOnboardingStore();

  return (
    <div style={styles.container}>
      <div style={styles.checkmark}>&#10003;</div>
      <h2 style={styles.title}>You're All Set!</h2>
      <p style={styles.description}>
        Your library is ready. AI DJ Assist will analyze your tracks in the background
        and start delivering intelligent recommendations.
      </p>
      <button style={styles.startBtn} onClick={completeOnboarding}>
        Start DJing
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
    maxWidth: 440,
  },
  checkmark: {
    width: 64,
    height: 64,
    borderRadius: "var(--radius-full)",
    backgroundColor: "var(--status-success)",
    color: "var(--text-inverse)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 32,
    fontWeight: 700,
    marginBottom: "var(--space-lg)",
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
  startBtn: {
    padding: "var(--space-sm) var(--space-xl)",
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    border: "none",
    borderRadius: "var(--radius-md)",
    fontSize: "var(--font-size-md)",
    fontWeight: 600,
    cursor: "pointer",
    transition: "background-color var(--transition-fast)",
  },
};
