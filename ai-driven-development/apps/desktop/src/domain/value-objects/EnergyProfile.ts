export type Trajectory = "build" | "maintain" | "drop";

interface EnergyProfileProps {
  overall: number;
  rmsValues?: readonly number[];
  spectralCentroid?: number;
}

export class EnergyProfile {
  public readonly overall: number;
  public readonly rmsValues: readonly number[] | undefined;
  public readonly spectralCentroid: number | undefined;

  private constructor(props: EnergyProfileProps) {
    this.overall = props.overall;
    this.rmsValues = props.rmsValues;
    this.spectralCentroid = props.spectralCentroid;
  }

  static create(props: EnergyProfileProps): EnergyProfile {
    if (props.overall < 0 || props.overall > 100) {
      throw new RangeError(`Energy must be 0-100, got ${props.overall}`);
    }
    return new EnergyProfile(props);
  }

  get trajectory(): Trajectory {
    if (!this.rmsValues || this.rmsValues.length < 2) return "maintain";

    const first = this.rmsValues[0] ?? 0;
    const last = this.rmsValues[this.rmsValues.length - 1] ?? 0;
    const diff = last - first;
    const threshold = 0.1;

    if (diff > threshold) return "build";
    if (diff < -threshold) return "drop";
    return "maintain";
  }

  compatibilityScore(other: EnergyProfile): number {
    const diff = Math.abs(this.overall - other.overall);
    return Math.max(0, 100 - diff * 2);
  }

  equals(other: EnergyProfile): boolean {
    return this.overall === other.overall;
  }
}
