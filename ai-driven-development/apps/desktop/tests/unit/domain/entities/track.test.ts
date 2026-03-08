import { describe, it, expect } from "vitest";
import { Track } from "@domain/entities/Track";
import { Recommendation } from "@domain/entities/Recommendation";
import { MixHistory } from "@domain/entities/MixHistory";

describe("Track", () => {
  it("should create a track with required properties", () => {
    const track = Track.create({
      id: "track-1",
      title: "Strobe",
      artist: "Deadmau5",
      filePath: "/music/strobe.mp3",
    });
    expect(track.id).toBe("track-1");
    expect(track.title).toBe("Strobe");
    expect(track.artist).toBe("Deadmau5");
    expect(track.filePath).toBe("/music/strobe.mp3");
  });

  it("should create a track with optional properties", () => {
    const track = Track.create({
      id: "track-2",
      title: "One More Time",
      artist: "Daft Punk",
      filePath: "/music/omt.mp3",
      album: "Discovery",
      durationMs: 320000,
      genre: "House",
    });
    expect(track.album).toBe("Discovery");
    expect(track.durationMs).toBe(320000);
    expect(track.genre).toBe("House");
  });

  it("should be equal by ID", () => {
    const a = Track.create({ id: "t1", title: "A", artist: "X", filePath: "/a" });
    const b = Track.create({ id: "t1", title: "B", artist: "Y", filePath: "/b" });
    expect(a.equals(b)).toBe(true);
  });

  it("should not be equal with different IDs", () => {
    const a = Track.create({ id: "t1", title: "A", artist: "X", filePath: "/a" });
    const b = Track.create({ id: "t2", title: "A", artist: "X", filePath: "/a" });
    expect(a.equals(b)).toBe(false);
  });
});

describe("Recommendation", () => {
  it("should create with valid score", () => {
    const rec = Recommendation.create({
      trackId: "track-1",
      score: 85,
      breakdown: {
        bpmScore: 90,
        keyScore: 80,
        energyScore: 85,
        genreScore: 75,
        historyScore: 90,
      },
    });
    expect(rec.trackId).toBe("track-1");
    expect(rec.score).toBe(85);
  });

  it("should reject score below 0", () => {
    expect(() =>
      Recommendation.create({
        trackId: "t1",
        score: -1,
        breakdown: { bpmScore: 0, keyScore: 0, energyScore: 0, genreScore: 0, historyScore: 0 },
      }),
    ).toThrow();
  });

  it("should reject score above 100", () => {
    expect(() =>
      Recommendation.create({
        trackId: "t1",
        score: 101,
        breakdown: {
          bpmScore: 100,
          keyScore: 100,
          energyScore: 100,
          genreScore: 100,
          historyScore: 100,
        },
      }),
    ).toThrow();
  });
});

describe("MixHistory", () => {
  it("should create with track pair", () => {
    const mix = MixHistory.create({
      id: "mix-1",
      fromTrackId: "track-1",
      toTrackId: "track-2",
      timestamp: new Date("2026-01-01"),
    });
    expect(mix.fromTrackId).toBe("track-1");
    expect(mix.toTrackId).toBe("track-2");
  });

  it("should associate two tracks", () => {
    const mix = MixHistory.create({
      id: "mix-1",
      fromTrackId: "track-1",
      toTrackId: "track-2",
      timestamp: new Date(),
    });
    expect(mix.involvesTracks("track-1", "track-2")).toBe(true);
    expect(mix.involvesTracks("track-2", "track-1")).toBe(true);
  });
});
