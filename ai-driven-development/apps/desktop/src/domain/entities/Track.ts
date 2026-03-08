interface TrackProps {
  id: string;
  title: string;
  artist: string;
  filePath: string;
  album?: string;
  durationMs?: number;
  genre?: string;
}

export class Track {
  readonly id: string;
  readonly title: string;
  readonly artist: string;
  readonly filePath: string;
  readonly album?: string;
  readonly durationMs?: number;
  readonly genre?: string;

  private constructor(props: TrackProps) {
    this.id = props.id;
    this.title = props.title;
    this.artist = props.artist;
    this.filePath = props.filePath;
    this.album = props.album;
    this.durationMs = props.durationMs;
    this.genre = props.genre;
  }

  static create(props: TrackProps): Track {
    return new Track(props);
  }

  equals(other: Track): boolean {
    return this.id === other.id;
  }
}
