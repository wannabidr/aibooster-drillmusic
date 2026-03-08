export type CamelotLetter = "A" | "B";

export class CamelotPosition {
  private constructor(
    public readonly number: number,
    public readonly letter: CamelotLetter,
  ) {}

  static create(notation: string): CamelotPosition {
    const match = notation.match(/^(\d{1,2})([AB])$/);
    if (!match) {
      throw new Error(`Invalid Camelot notation: ${notation}`);
    }
    const num = parseInt(match[1] ?? "", 10);
    const letter = match[2] as CamelotLetter;
    if (num < 1 || num > 12) {
      throw new RangeError(`Camelot number must be 1-12, got ${num}`);
    }
    return new CamelotPosition(num, letter);
  }

  adjacentKeys(): CamelotPosition[] {
    const prev = this.number === 1 ? 12 : this.number - 1;
    const next = this.number === 12 ? 1 : this.number + 1;
    const opposite: CamelotLetter = this.letter === "A" ? "B" : "A";

    return [
      new CamelotPosition(prev, this.letter),
      new CamelotPosition(next, this.letter),
      new CamelotPosition(this.number, opposite),
    ];
  }

  compatibilityScore(other: CamelotPosition): number {
    if (this.equals(other)) return 100;

    const numDiff = this.circularDistance(other.number);

    // Same number, different letter = energy boost
    if (numDiff === 0 && this.letter !== other.letter) return 85;

    // Adjacent on wheel (same letter, +-1)
    if (numDiff === 1 && this.letter === other.letter) return 90;

    // Adjacent number + mode switch
    if (numDiff === 1 && this.letter !== other.letter) return 75;

    // 2 steps same letter
    if (numDiff === 2 && this.letter === other.letter) return 60;

    // Everything else degrades
    return Math.max(0, 100 - numDiff * 15 - (this.letter !== other.letter ? 10 : 0));
  }

  private circularDistance(otherNum: number): number {
    const diff = Math.abs(this.number - otherNum);
    return Math.min(diff, 12 - diff);
  }

  equals(other: CamelotPosition): boolean {
    return this.number === other.number && this.letter === other.letter;
  }

  toString(): string {
    return `${this.number}${this.letter}`;
  }
}
