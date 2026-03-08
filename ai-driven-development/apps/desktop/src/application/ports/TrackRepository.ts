import { Track } from "@domain/entities/Track";

export interface TrackRepository {
  findById(id: string): Track | undefined;
  findAll(): Track[];
  save(track: Track): void;
}
