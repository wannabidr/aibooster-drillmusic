import { TrackAnalysisData } from "../dto/RecommendationDTO";

export interface AnalysisDataProvider {
  getAnalysisData(trackId: string): TrackAnalysisData | undefined;
  getAllAnalysisData(): TrackAnalysisData[];
}
