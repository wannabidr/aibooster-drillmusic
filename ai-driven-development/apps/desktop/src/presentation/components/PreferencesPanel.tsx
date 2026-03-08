import {
  usePreferencesStore,
  type EnergyDirection,
  type BlendStyle,
  type WeightKey,
} from "../../infrastructure/state/usePreferencesStore";

const WEIGHT_LABELS: { key: WeightKey; label: string }[] = [
  { key: "bpm", label: "BPM" },
  { key: "key", label: "Key" },
  { key: "energy", label: "Energy" },
  { key: "genre", label: "Genre" },
  { key: "history", label: "History" },
];

const ENERGY_DIRECTIONS: { value: EnergyDirection; label: string }[] = [
  { value: "build", label: "Build" },
  { value: "maintain", label: "Maintain" },
  { value: "drop", label: "Drop" },
];

const BLEND_STYLES: { value: BlendStyle; label: string }[] = [
  { value: "long_blend", label: "Long Blend" },
  { value: "short_cut", label: "Short Cut" },
  { value: "echo_out", label: "Echo Out" },
  { value: "filter_sweep", label: "Filter Sweep" },
  { value: "backspin", label: "Backspin" },
];

export function PreferencesPanel() {
  const { weights, energyDirection, blendStyle, setWeight, setEnergyDirection, setBlendStyle } =
    usePreferencesStore();

  return (
    <div style={styles.container}>
      <h3 style={styles.sectionTitle}>Signal Weights</h3>
      <div style={styles.sliderGroup}>
        {WEIGHT_LABELS.map(({ key, label }) => (
          <div key={key} style={styles.sliderRow}>
            <label htmlFor={`weight-${key}`} style={styles.sliderLabel}>
              {label}
            </label>
            <input
              id={`weight-${key}`}
              aria-label={label}
              type="range"
              min={0}
              max={100}
              value={weights[key]}
              onChange={(e) => setWeight(key, Number(e.target.value))}
              style={styles.slider}
            />
            <span style={styles.sliderValue}>{weights[key]}</span>
          </div>
        ))}
      </div>

      <h3 style={styles.sectionTitle}>Energy Direction</h3>
      <div style={styles.buttonGroup}>
        {ENERGY_DIRECTIONS.map(({ value, label }) => (
          <button
            key={value}
            aria-label={label}
            data-active={String(energyDirection === value)}
            onClick={() => setEnergyDirection(value)}
            style={{
              ...styles.optionButton,
              ...(energyDirection === value ? styles.optionButtonActive : {}),
            }}
          >
            {label}
          </button>
        ))}
      </div>

      <h3 style={styles.sectionTitle}>Default Blend Style</h3>
      <div style={styles.buttonGroup}>
        {BLEND_STYLES.map(({ value, label }) => (
          <button
            key={value}
            aria-label={label}
            data-active={String(blendStyle === value)}
            onClick={() => setBlendStyle(value)}
            style={{
              ...styles.optionButton,
              ...(blendStyle === value ? styles.optionButtonActive : {}),
            }}
          >
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "var(--space-lg)",
    padding: "var(--space-md)",
  },
  sectionTitle: {
    fontSize: "var(--font-size-sm)",
    fontWeight: 600,
    color: "var(--text-secondary)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: "var(--space-xs)",
  },
  sliderGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "var(--space-sm)",
  },
  sliderRow: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-sm)",
  },
  sliderLabel: {
    width: 60,
    fontSize: "var(--font-size-sm)",
    color: "var(--text-primary)",
  },
  slider: {
    flex: 1,
    accentColor: "var(--accent-primary)",
  },
  sliderValue: {
    width: 32,
    textAlign: "right",
    fontSize: "var(--font-size-xs)",
    fontFamily: "var(--font-mono)",
    color: "var(--text-muted)",
  },
  buttonGroup: {
    display: "flex",
    gap: "var(--space-xs)",
    flexWrap: "wrap",
  },
  optionButton: {
    padding: "var(--space-xs) var(--space-sm)",
    backgroundColor: "var(--bg-elevated)",
    color: "var(--text-secondary)",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "var(--border-subtle)",
    borderRadius: "var(--radius-sm)",
    cursor: "pointer",
    fontSize: "var(--font-size-xs)",
    fontWeight: 500,
    transition: "all var(--transition-fast)",
  },
  optionButtonActive: {
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    borderColor: "var(--accent-primary)",
  },
};
