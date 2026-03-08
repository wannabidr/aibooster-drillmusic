interface MixHistoryProps {
  id: string;
  fromTrackId: string;
  toTrackId: string;
  timestamp: Date;
}

export class MixHistory {
  readonly id: string;
  readonly fromTrackId: string;
  readonly toTrackId: string;
  readonly timestamp: Date;

  private constructor(props: MixHistoryProps) {
    this.id = props.id;
    this.fromTrackId = props.fromTrackId;
    this.toTrackId = props.toTrackId;
    this.timestamp = props.timestamp;
  }

  static create(props: MixHistoryProps): MixHistory {
    return new MixHistory(props);
  }

  involvesTracks(trackA: string, trackB: string): boolean {
    return (
      (this.fromTrackId === trackA && this.toTrackId === trackB) ||
      (this.fromTrackId === trackB && this.toTrackId === trackA)
    );
  }

  equals(other: MixHistory): boolean {
    return this.id === other.id;
  }
}
