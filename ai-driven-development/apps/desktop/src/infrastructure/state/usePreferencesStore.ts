import { create } from "zustand";

export type EnergyDirection = "build" | "maintain" | "drop";
export type BlendStyle =
  | "long_blend"
  | "short_cut"
  | "echo_out"
  | "filter_sweep"
  | "backspin";
export type WeightKey = "bpm" | "key" | "energy" | "genre" | "history";

export interface SignalWeights {
  bpm: number;
  key: number;
  energy: number;
  genre: number;
  history: number;
}

const DEFAULT_WEIGHTS: SignalWeights = {
  bpm: 20,
  key: 25,
  energy: 25,
  genre: 15,
  history: 15,
};

interface PreferencesState {
  weights: SignalWeights;
  energyDirection: EnergyDirection;
  blendStyle: BlendStyle;
  setWeight: (key: WeightKey, value: number) => void;
  setEnergyDirection: (direction: EnergyDirection) => void;
  setBlendStyle: (style: BlendStyle) => void;
  reset: () => void;
  getNormalizedWeights: () => SignalWeights;
}

export const usePreferencesStore = create<PreferencesState>((set, get) => ({
  weights: { ...DEFAULT_WEIGHTS },
  energyDirection: "maintain",
  blendStyle: "long_blend",

  setWeight: (key, value) =>
    set((s) => ({
      weights: {
        ...s.weights,
        [key]: Math.max(0, Math.min(100, value)),
      },
    })),

  setEnergyDirection: (direction) => set({ energyDirection: direction }),

  setBlendStyle: (style) => set({ blendStyle: style }),

  reset: () =>
    set({
      weights: { ...DEFAULT_WEIGHTS },
      energyDirection: "maintain",
      blendStyle: "long_blend",
    }),

  getNormalizedWeights: () => {
    const { weights } = get();
    const sum = weights.bpm + weights.key + weights.energy + weights.genre + weights.history;
    if (sum === 0) {
      return { bpm: 0.2, key: 0.2, energy: 0.2, genre: 0.2, history: 0.2 };
    }
    return {
      bpm: weights.bpm / sum,
      key: weights.key / sum,
      energy: weights.energy / sum,
      genre: weights.genre / sum,
      history: weights.history / sum,
    };
  },
}));
