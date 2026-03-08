import { create } from "zustand";

export type ImportSource = "rekordbox" | "serato" | "traktor" | "folder";

const TOTAL_STEPS = 6;

interface OnboardingState {
  currentStep: number;
  totalSteps: number;
  completed: boolean;
  importSource: ImportSource | null;
  historyOptIn: boolean;
  communityOptIn: boolean;

  nextStep: () => void;
  prevStep: () => void;
  goToStep: (step: number) => void;
  completeOnboarding: () => void;
  setImportSource: (source: ImportSource) => void;
  setHistoryOptIn: (optIn: boolean) => void;
  setCommunityOptIn: (optIn: boolean) => void;
  reset: () => void;
}

export const useOnboardingStore = create<OnboardingState>((set) => ({
  currentStep: 0,
  totalSteps: TOTAL_STEPS,
  completed: false,
  importSource: null,
  historyOptIn: false,
  communityOptIn: false,

  nextStep: () =>
    set((s) => ({ currentStep: Math.min(s.currentStep + 1, TOTAL_STEPS - 1) })),
  prevStep: () =>
    set((s) => ({ currentStep: Math.max(s.currentStep - 1, 0) })),
  goToStep: (step) =>
    set(() => ({ currentStep: Math.max(0, Math.min(step, TOTAL_STEPS - 1)) })),
  completeOnboarding: () => set({ completed: true }),
  setImportSource: (source) => set({ importSource: source }),
  setHistoryOptIn: (optIn) => set({ historyOptIn: optIn }),
  setCommunityOptIn: (optIn) => set({ communityOptIn: optIn }),
  reset: () =>
    set({
      currentStep: 0,
      completed: false,
      importSource: null,
      historyOptIn: false,
      communityOptIn: false,
    }),
}));
