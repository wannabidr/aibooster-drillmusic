import { create } from "zustand";

interface ThemeState {
  highContrast: boolean;
  performanceMode: boolean;
  toggleHighContrast: () => void;
  togglePerformanceMode: () => void;
  setHighContrast: (value: boolean) => void;
  setPerformanceMode: (value: boolean) => void;
}

export const useThemeStore = create<ThemeState>((set) => ({
  highContrast: false,
  performanceMode: false,
  toggleHighContrast: () => set((s) => ({ highContrast: !s.highContrast })),
  togglePerformanceMode: () =>
    set((s) => ({ performanceMode: !s.performanceMode })),
  setHighContrast: (value) => set({ highContrast: value }),
  setPerformanceMode: (value) => set({ performanceMode: value }),
}));
