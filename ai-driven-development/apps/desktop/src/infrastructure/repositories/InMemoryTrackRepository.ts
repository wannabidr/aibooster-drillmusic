import { Track } from "@domain/entities/Track";
import { TrackRepository } from "@application/ports/TrackRepository";

export class InMemoryTrackRepository implements TrackRepository {
  private tracks = new Map<string, Track>();

  findById(id: string): Track | undefined {
    return this.tracks.get(id);
  }

  findAll(): Track[] {
    return Array.from(this.tracks.values());
  }

  save(track: Track): void {
    this.tracks.set(track.id, track);
  }
}
