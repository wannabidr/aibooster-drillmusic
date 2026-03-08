export interface ScoreBreakdown {
  bpmScore: number;
  keyScore: number;
  energyScore: number;
  genreScore: number;
  historyScore: number;
}

interface RecommendationProps {
  trackId: string;
  score: number;
  breakdown: ScoreBreakdown;
  reason?: string;
}

export class Recommendation {
  readonly trackId: string;
  readonly score: number;
  readonly breakdown: ScoreBreakdown;
  readonly reason?: string;

  private constructor(props: RecommendationProps) {
    this.trackId = props.trackId;
    this.score = props.score;
    this.breakdown = props.breakdown;
    this.reason = props.reason;
  }

  static create(props: RecommendationProps): Recommendation {
    if (props.score < 0 || props.score > 100) {
      throw new RangeError(`Score must be 0-100, got ${props.score}`);
    }
    return new Recommendation(props);
  }
}
