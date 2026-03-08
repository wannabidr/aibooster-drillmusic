import { useOnboardingStore } from "../../../infrastructure/state/useOnboardingStore";

export function WelcomeScreen() {
  const { nextStep } = useOnboardingStore();

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Welcome to AI DJ Assist</h1>
      <p style={styles.subtitle}>
        Your intelligent mixing companion for professional club and festival DJing.
      </p>
      <div style={styles.features}>
        <Feature
          title="Smart Recommendations"
          description="AI-powered next-track suggestions based on BPM, key, energy, and genre."
        />
        <Feature
          title="Harmonic Mixing"
          description="Camelot wheel integration ensures every transition sounds musical."
        />
        <Feature
          title="Audio Preview"
          description="Hear AI-rendered blend previews before committing to a transition."
        />
      </div>
      <button style={styles.startBtn} onClick={nextStep}>
        Get Started
      </button>
    </div>
  );
}

function Feature({ title, description }: { title: string; description: string }) {
  return (
    <div style={styles.feature}>
      <h3 style={styles.featureTitle}>{title}</h3>
      <p style={styles.featureDesc}>{description}</p>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    textAlign: "center",
    maxWidth: 560,
  },
  title: {
    fontSize: "var(--font-size-2xl)",
    fontWeight: 700,
    marginBottom: "var(--space-md)",
  },
  subtitle: {
    fontSize: "var(--font-size-md)",
    color: "var(--text-secondary)",
    marginBottom: "var(--space-xl)",
    lineHeight: 1.6,
  },
  features: {
    display: "flex",
    flexDirection: "column",
    gap: "var(--space-md)",
    marginBottom: "var(--space-xl)",
    width: "100%",
  },
  feature: {
    padding: "var(--space-md)",
    backgroundColor: "var(--bg-surface)",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--border-subtle)",
    textAlign: "left",
  },
  featureTitle: {
    fontSize: "var(--font-size-sm)",
    fontWeight: 600,
    marginBottom: "var(--space-xs)",
  },
  featureDesc: {
    fontSize: "var(--font-size-xs)",
    color: "var(--text-secondary)",
    lineHeight: 1.5,
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
