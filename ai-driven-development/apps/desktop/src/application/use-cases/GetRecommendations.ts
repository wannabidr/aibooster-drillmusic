import { BPM } from "@domain/value-objects/BPM";
import { CamelotPosition } from "@domain/value-objects/CamelotPosition";
import { EnergyProfile } from "@domain/value-objects/EnergyProfile";
import { RecommendationScorer } from "@domain/services/RecommendationScorer";
import { computeGenreScore } from "@domain/services/GenreSimilarity";
import { AnalysisDataProvider } from "../ports/AnalysisDataProvider";
import { ScoringPort } from "../ports/ScoringPort";
import {
  RecommendationRequest,
  RecommendationResponse,
  TrackAnalysisData,
} from "../dto/RecommendationDTO";

const DEFAULT_LIMIT = 10;
const NEUTRAL_SCORE = 50;

export class GetRecommendations {
  private readonly scorer = new RecommendationScorer();

  constructor(
    private readonly analysisProvider: AnalysisDataProvider,
    private readonly mlScorer?: ScoringPort,
  ) {}

  execute(request: RecommendationRequest): RecommendationResponse[] {
    const limit = request.limit ?? DEFAULT_LIMIT;
    const excludeSet = new Set(request.excludeTrackIds ?? []);
    excludeSet.add(request.currentTrackId);

    const currentData = this.analysisProvider.getAnalysisData(request.currentTrackId);
    if (!currentData) return [];

    const candidates = this.analysisProvider
      .getAllAnalysisData()
      .filter((d) => !excludeSet.has(d.trackId));

    const scored =
      this.mlScorer?.isAvailable
        ? this.scoreCandidatesML(currentData, candidates)
        : candidates.map((candidate) =>
            this.scoreCandidate(currentData, candidate),
          );

    scored.sort((a, b) => b.score - a.score);

    const topN = scored.slice(0, limit);
    const confidence = this.computeConfidence(candidates.length, topN);

    return topN.map((r) => ({ ...r, confidence }));
  }

  private scoreCandidate(
    current: TrackAnalysisData,
    candidate: TrackAnalysisData,
  ): RecommendationResponse {
    const bpmScore = this.computeBpmScore(current.bpm, candidate.bpm);
    const keyScore = this.computeKeyScore(current.camelotPosition, candidate.camelotPosition);
    const energyScore = this.computeEnergyScore(current.energy, candidate.energy);
    const genreScore = computeGenreScore(current.genreEmbedding, candidate.genreEmbedding);
    const historyScore = NEUTRAL_SCORE;

    const breakdown = { bpmScore, keyScore, energyScore, genreScore, historyScore };
    const score = this.scorer.computeScore(breakdown);

    return {
      trackId: candidate.trackId,
      score,
      breakdown,
      confidence: 0,
    };
  }

  private computeBpmScore(currentBpm: number, candidateBpm: number): number {
    try {
      const a = BPM.create(currentBpm);
      const b = BPM.create(candidateBpm);
      return a.compatibilityScore(b);
    } catch {
      return 0;
    }
  }

  private computeKeyScore(currentKey: string, candidateKey: string): number {
    try {
      const a = CamelotPosition.create(currentKey);
      const b = CamelotPosition.create(candidateKey);
      return a.compatibilityScore(b);
    } catch {
      return 0;
    }
  }

  private computeEnergyScore(currentEnergy: number, candidateEnergy: number): number {
    const a = EnergyProfile.create({ overall: currentEnergy });
    const b = EnergyProfile.create({ overall: candidateEnergy });
    return a.compatibilityScore(b);
  }

  private scoreCandidatesML(
    current: TrackAnalysisData,
    candidates: TrackAnalysisData[],
  ): RecommendationResponse[] {
    const mlScores = this.mlScorer!.scoreBatch(current, candidates);
    return candidates.map((candidate, i) => ({
      trackId: candidate.trackId,
      score: mlScores[i]!,
      breakdown: {
        bpmScore: mlScores[i]!,
        keyScore: mlScores[i]!,
        energyScore: mlScores[i]!,
        genreScore: mlScores[i]!,
        historyScore: mlScores[i]!,
      },
      confidence: 0,
    }));
  }

  private computeConfidence(candidateCount: number, topN: RecommendationResponse[]): number {
    if (candidateCount === 0 || topN.length === 0) return 0;

    const countFactor = Math.min(1, candidateCount / 20);
    const avgScore = topN.reduce((sum, r) => sum + r.score, 0) / topN.length;
    const scoreFactor = avgScore / 100;

    return Math.round((countFactor * 0.4 + scoreFactor * 0.6) * 100);
  }
}
