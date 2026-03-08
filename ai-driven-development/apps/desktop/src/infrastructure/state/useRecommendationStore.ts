import { create } from "zustand";
import { ScoreBreakdown } from "@domain/entities/Recommendation";

export interface RecommendationItem {
  trackId: string;
  score: number;
  breakdown: ScoreBreakdown;
  confidence: number;
  reason?: string;
}

interface RecommendationState {
  recommendations: RecommendationItem[];
  isLoading: boolean;
  confidence: number;
  error: string | null;

  setRecommendations: (recs: RecommendationItem[], confidence: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clear: () => void;
}

export const useRecommendationStore = create<RecommendationState>((set) => ({
  recommendations: [],
  isLoading: false,
  confidence: 0,
  error: null,

  setRecommendations: (recommendations, confidence) =>
    set({ recommendations, confidence, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
  clear: () => set({ recommendations: [], confidence: 0, error: null }),
}));
