import { ScoreBreakdown } from "@domain/entities/Recommendation";

export interface RecommendationRequest {
  currentTrackId: string;
  limit?: number;
  excludeTrackIds?: string[];
}

export interface RecommendationResponse {
  trackId: string;
  score: number;
  breakdown: ScoreBreakdown;
  confidence: number;
  reason?: string;
}

export interface TrackAnalysisData {
  trackId: string;
  bpm: number;
  camelotPosition: string;
  energy: number;
  genre?: string;
  genreEmbedding?: number[];
}
