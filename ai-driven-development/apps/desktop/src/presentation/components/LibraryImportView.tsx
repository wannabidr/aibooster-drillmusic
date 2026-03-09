type ImportSource = "rekordbox" | "serato" | "traktor" | "folder";

interface LibraryImportViewProps {
  isImporting: boolean;
  importProgress?: { current: number; total: number; source: ImportSource };
  onImport: (source: ImportSource) => void;
}

export function LibraryImportView({
  isImporting,
  importProgress,
  onImport,
}: LibraryImportViewProps) {
  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Import Library</h2>
      <p style={styles.description}>
        Import your track library from DJ software or a folder.
      </p>

      <div style={styles.sources}>
        <ImportButton
          label="Rekordbox"
          source="rekordbox"
          disabled={isImporting}
          onClick={onImport}
        />
        <ImportButton
          label="Serato DJ"
          source="serato"
          disabled={isImporting}
          onClick={onImport}
        />
        <ImportButton
          label="Traktor"
          source="traktor"
          disabled={isImporting}
          onClick={onImport}
        />
        <ImportButton
          label="Folder"
          source="folder"
          disabled={isImporting}
          onClick={onImport}
        />
      </div>

      {isImporting && importProgress && (
        <div style={styles.progress}>
          <div style={styles.progressLabel}>
            Importing from {importProgress.source}...
            {importProgress.current} / {importProgress.total}
          </div>
          <div style={styles.progressTrack}>
            <div
              style={{
                ...styles.progressBar,
                width: `${(importProgress.current / importProgress.total) * 100}%`,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function ImportButton({
  label,
  source,
  disabled,
  onClick,
}: {
  label: string;
  source: ImportSource;
  disabled: boolean;
  onClick: (source: ImportSource) => void;
}) {
  return (
    <button
      style={{
        ...styles.importBtn,
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
      }}
      disabled={disabled}
      onClick={() => {
        console.log(`[LibraryImportView] Button clicked: ${source}`);
        onClick(source);
      }}
    >
      {label}
    </button>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    padding: "var(--space-2xl)",
    textAlign: "center",
  },
  title: {
    fontSize: "var(--font-size-xl)",
    fontWeight: 600,
    marginBottom: "var(--space-sm)",
  },
  description: {
    color: "var(--text-secondary)",
    marginBottom: "var(--space-xl)",
  },
  sources: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "var(--space-md)",
    maxWidth: 400,
    width: "100%",
  },
  importBtn: {
    padding: "var(--space-lg) var(--space-md)",
    backgroundColor: "var(--bg-elevated)",
    color: "var(--text-primary)",
    border: "1px solid var(--border-default)",
    borderRadius: "var(--radius-md)",
    fontSize: "var(--font-size-md)",
    fontWeight: 500,
    transition: "all var(--transition-fast)",
  },
  progress: {
    marginTop: "var(--space-xl)",
    width: "100%",
    maxWidth: 400,
  },
  progressLabel: {
    fontSize: "var(--font-size-sm)",
    color: "var(--text-secondary)",
    marginBottom: "var(--space-sm)",
  },
  progressTrack: {
    height: 6,
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
};
