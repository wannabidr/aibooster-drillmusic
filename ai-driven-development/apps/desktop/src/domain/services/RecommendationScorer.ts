import { ScoreBreakdown } from "../entities/Recommendation";

const WEIGHTS = {
  bpm: 0.2,
  key: 0.25,
  energy: 0.25,
  genre: 0.15,
  history: 0.15,
} as const;

export class RecommendationScorer {
  computeScore(breakdown: ScoreBreakdown): number {
    const weighted =
      breakdown.bpmScore * WEIGHTS.bpm +
      breakdown.keyScore * WEIGHTS.key +
      breakdown.energyScore * WEIGHTS.energy +
      breakdown.genreScore * WEIGHTS.genre +
      breakdown.historyScore * WEIGHTS.history;

    return Math.round(Math.max(0, Math.min(100, weighted)));
  }
}
