import { AnalysisDataProvider } from "@application/ports/AnalysisDataProvider";
import { TrackAnalysisData } from "@application/dto/RecommendationDTO";

export class InMemoryAnalysisDataProvider implements AnalysisDataProvider {
  private data = new Map<string, TrackAnalysisData>();

  add(analysisData: TrackAnalysisData): void {
    this.data.set(analysisData.trackId, analysisData);
  }

  getAnalysisData(trackId: string): TrackAnalysisData | undefined {
    return this.data.get(trackId);
  }

  getAllAnalysisData(): TrackAnalysisData[] {
    return Array.from(this.data.values());
  }
}
