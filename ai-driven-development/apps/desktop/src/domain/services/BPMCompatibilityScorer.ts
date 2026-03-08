import { BPM } from "../value-objects/BPM";

export class BPMCompatibilityScorer {
  score(a: BPM, b: BPM): number {
    return a.compatibilityScore(b);
  }
}
