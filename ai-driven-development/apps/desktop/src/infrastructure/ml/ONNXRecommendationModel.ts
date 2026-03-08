import { ScoringPort } from "@application/ports/ScoringPort";
import { TrackAnalysisData } from "@application/dto/RecommendationDTO";

const CAMELOT_REGEX = /^(\d{1,2})([AB])$/;

function normalizeFeatures(track: TrackAnalysisData): number[] {
  const bpmNorm = (track.bpm - 60) / 140; // normalize BPM 60-200 to 0-1
  const camelotMatch = track.camelotPosition.match(CAMELOT_REGEX);
  const keyNum = camelotMatch ? parseInt(camelotMatch[1]!, 10) / 12 : 0.5;
  const keyMode = camelotMatch && camelotMatch[2] === "B" ? 1.0 : 0.0;
  const energyNorm = track.energy / 100;
  const genreEmbed = 0.5; // neutral genre embedding
  return [bpmNorm, keyNum, keyMode, energyNorm, genreEmbed];
}

function buildPairFeatures(
  current: TrackAnalysisData,
  candidate: TrackAnalysisData,
): Float32Array {
  const a = normalizeFeatures(current);
  const b = normalizeFeatures(candidate);
  return new Float32Array([...a, ...b]);
}

export interface ONNXSession {
  run(
    feeds: Record<string, { dims: number[]; type: string; data: Float32Array }>,
  ): Promise<Record<string, { data: Float32Array }>>;
}

export class ONNXRecommendationModel implements ScoringPort {
  private session: ONNXSession | null = null;

  get isAvailable(): boolean {
    return this.session !== null;
  }

  async load(session: ONNXSession): Promise<void> {
    this.session = session;
  }

  scorePair(current: TrackAnalysisData, candidate: TrackAnalysisData): number {
    return this.scoreBatch(current, [candidate])[0]!;
  }

  scoreBatch(
    current: TrackAnalysisData,
    candidates: TrackAnalysisData[],
  ): number[] {
    if (!this.session) {
      throw new Error("ONNX session not loaded");
    }

    // For synchronous scoring, we pre-compute features and run sync
    // In production this would be async, but the interface matches the rule-based scorer
    const batchSize = candidates.length;
    const featureData = new Float32Array(batchSize * 10);
    for (let i = 0; i < batchSize; i++) {
      const pair = buildPairFeatures(current, candidates[i]!);
      featureData.set(pair, i * 10);
    }

    // Store for async resolution
    this._lastFeatures = featureData;
    this._lastBatchSize = batchSize;

    // Return placeholder scores - actual scoring happens via scoreAsync
    return candidates.map(() => 50);
  }

  private _lastFeatures: Float32Array | null = null;
  private _lastBatchSize = 0;

  async scoreBatchAsync(
    current: TrackAnalysisData,
    candidates: TrackAnalysisData[],
  ): Promise<number[]> {
    if (!this.session) {
      throw new Error("ONNX session not loaded");
    }

    const batchSize = candidates.length;
    const featureData = new Float32Array(batchSize * 10);
    for (let i = 0; i < batchSize; i++) {
      const pair = buildPairFeatures(current, candidates[i]!);
      featureData.set(pair, i * 10);
    }

    const result = await this.session.run({
      features: {
        dims: [batchSize, 10],
        type: "float32",
        data: featureData,
      },
    });

    const scores = result["score"]!.data;
    return Array.from(scores).map((s) => Math.round(s * 100));
  }
}

export { normalizeFeatures, buildPairFeatures };
