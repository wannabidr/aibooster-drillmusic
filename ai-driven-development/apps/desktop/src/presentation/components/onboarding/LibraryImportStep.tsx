import { useOnboardingStore, type ImportSource } from "../../../infrastructure/state/useOnboardingStore";

const SOURCES: { value: ImportSource; label: string }[] = [
  { value: "rekordbox", label: "Rekordbox" },
  { value: "serato", label: "Serato DJ" },
  { value: "traktor", label: "Traktor" },
  { value: "folder", label: "Music Folder" },
];

export function LibraryImportStep() {
  const { importSource, setImportSource, nextStep } = useOnboardingStore();

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Import Your Library</h2>
      <p style={styles.description}>
        Select your DJ software to import your track library. We'll analyze BPM, key, and energy for every track.
      </p>
      <div style={styles.sources}>
        {SOURCES.map(({ value, label }) => (
          <button
            key={value}
            style={{
              ...styles.sourceBtn,
              ...(importSource === value ? styles.sourceBtnActive : {}),
            }}
            onClick={() => setImportSource(value)}
          >
            {label}
          </button>
        ))}
      </div>
      <button
        style={{
          ...styles.continueBtn,
          opacity: importSource ? 1 : 0.5,
          cursor: importSource ? "pointer" : "not-allowed",
        }}
        disabled={!importSource}
        onClick={nextStep}
      >
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
  sources: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "var(--space-md)",
    width: "100%",
    marginBottom: "var(--space-xl)",
  },
  sourceBtn: {
    padding: "var(--space-lg) var(--space-md)",
    backgroundColor: "var(--bg-elevated)",
    color: "var(--text-primary)",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "var(--border-default)",
    borderRadius: "var(--radius-md)",
    fontSize: "var(--font-size-md)",
    fontWeight: 500,
    cursor: "pointer",
    transition: "all var(--transition-fast)",
  },
  sourceBtnActive: {
    borderColor: "var(--accent-primary)",
    backgroundColor: "var(--bg-active)",
  },
  continueBtn: {
    padding: "var(--space-sm) var(--space-xl)",
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    border: "none",
    borderRadius: "var(--radius-md)",
    fontSize: "var(--font-size-md)",
    fontWeight: 600,
    transition: "background-color var(--transition-fast)",
  },
};
