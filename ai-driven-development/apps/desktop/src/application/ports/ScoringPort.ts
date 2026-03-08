import { TrackAnalysisData } from "../dto/RecommendationDTO";

export interface ScoringPort {
  scorePair(current: TrackAnalysisData, candidate: TrackAnalysisData): number;
  scoreBatch(
    current: TrackAnalysisData,
    candidates: TrackAnalysisData[],
  ): number[];
  readonly isAvailable: boolean;
}
