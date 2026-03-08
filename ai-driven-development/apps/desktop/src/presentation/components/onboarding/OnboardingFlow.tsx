import { useOnboardingStore } from "../../../infrastructure/state/useOnboardingStore";
import { WelcomeScreen } from "./WelcomeScreen";
import { LibraryImportStep } from "./LibraryImportStep";
import { HistoryImportStep } from "./HistoryImportStep";
import { CommunityOptInStep } from "./CommunityOptInStep";
import { TutorialStep } from "./TutorialStep";
import { ReadyScreen } from "./ReadyScreen";

const STEPS = [
  WelcomeScreen,
  LibraryImportStep,
  HistoryImportStep,
  CommunityOptInStep,
  TutorialStep,
  ReadyScreen,
] as const;

export function OnboardingFlow() {
  const { currentStep, totalSteps, prevStep } = useOnboardingStore();

  const StepComponent = STEPS[currentStep];

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <StepComponent />
      </div>
      <div style={styles.footer}>
        {currentStep > 0 && (
          <button style={styles.backBtn} onClick={prevStep}>
            Back
          </button>
        )}
        <StepIndicator current={currentStep} total={totalSteps} />
      </div>
    </div>
  );
}

function StepIndicator({ current, total }: { current: number; total: number }) {
  return (
    <div data-testid="step-indicator" style={styles.indicator}>
      {Array.from({ length: total }, (_, i) => (
        <div
          key={i}
          style={{
            ...styles.dot,
            backgroundColor:
              i === current
                ? "var(--accent-primary)"
                : i < current
                  ? "var(--accent-secondary)"
                  : "var(--bg-overlay)",
          }}
        />
      ))}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    backgroundColor: "var(--bg-base)",
    color: "var(--text-primary)",
  },
  content: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "var(--space-2xl)",
    overflow: "auto",
  },
  footer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "var(--space-lg)",
    padding: "var(--space-lg)",
    borderTop: "1px solid var(--border-subtle)",
  },
  backBtn: {
    position: "absolute",
    left: "var(--space-lg)",
    padding: "var(--space-xs) var(--space-md)",
    backgroundColor: "transparent",
    color: "var(--text-secondary)",
    border: "1px solid var(--border-default)",
    borderRadius: "var(--radius-sm)",
    cursor: "pointer",
    fontSize: "var(--font-size-sm)",
  },
  indicator: {
    display: "flex",
    gap: "var(--space-sm)",
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: "var(--radius-full)",
    transition: "background-color var(--transition-normal)",
  },
};
