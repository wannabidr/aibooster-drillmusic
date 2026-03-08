import { CamelotPosition } from "../value-objects/CamelotPosition";

export class HarmonicCompatibilityCalculator {
  score(a: CamelotPosition, b: CamelotPosition): number {
    return a.compatibilityScore(b);
  }
}
