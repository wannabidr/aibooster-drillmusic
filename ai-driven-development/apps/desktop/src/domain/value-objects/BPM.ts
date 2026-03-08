export class BPM {
  private constructor(public readonly value: number) {}

  static create(value: number): BPM {
    if (value < 20 || value > 300) {
      throw new RangeError(`BPM must be between 20 and 300, got ${value}`);
    }
    return new BPM(value);
  }

  static unsafe(value: number): BPM {
    return new BPM(value);
  }

  halfTime(): BPM {
    return BPM.unsafe(this.value / 2);
  }

  doubleTime(): BPM {
    return BPM.unsafe(this.value * 2);
  }

  compatibilityScore(other: BPM): number {
    const diff = Math.abs(this.value - other.value);
    const halfDiff = Math.abs(this.value - other.value * 2);
    const doubleDiff = Math.abs(this.value * 2 - other.value);

    const minDiff = Math.min(diff, halfDiff, doubleDiff);

    if (minDiff === 0) return 100;
    if (minDiff <= 2) return 95;
    if (minDiff <= 4) return 85;
    if (minDiff <= 8) return 70;
    if (minDiff <= 16) return 40;
    return Math.max(0, 100 - minDiff * 3);
  }

  equals(other: BPM): boolean {
    return this.value === other.value;
  }
}
