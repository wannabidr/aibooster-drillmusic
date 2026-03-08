import { describe, it, expect, beforeEach } from "vitest";
import { useLibraryStore } from "@infrastructure/state/useLibraryStore";
import { useRecommendationStore } from "@infrastructure/state/useRecommendationStore";
import { usePlayerStore } from "@infrastructure/state/usePlayerStore";

describe("useLibraryStore", () => {
  beforeEach(() => {
    useLibraryStore.setState({
      tracks: [],
      selectedTrackId: null,
      isImporting: false,
      importProgress: null,
      searchQuery: "",
    });
  });

  it("should add and retrieve tracks", () => {
    const store = useLibraryStore.getState();
    store.setTracks([
      { id: "t1", title: "Track 1", artist: "Artist 1", filePath: "/a.mp3", analyzed: false },
    ]);
    expect(useLibraryStore.getState().tracks).toHaveLength(1);
  });

  it("should select a track", () => {
    const store = useLibraryStore.getState();
    store.selectTrack("t1");
    expect(useLibraryStore.getState().selectedTrackId).toBe("t1");
  });

  it("should update a track", () => {
    const store = useLibraryStore.getState();
    store.setTracks([
      { id: "t1", title: "Track", artist: "Artist", filePath: "/a.mp3", analyzed: false },
    ]);
    store.updateTrack("t1", { bpm: 128, analyzed: true });
    const track = useLibraryStore.getState().tracks[0]!;
    expect(track.bpm).toBe(128);
    expect(track.analyzed).toBe(true);
  });

  it("should track import progress", () => {
    const store = useLibraryStore.getState();
    store.setImporting(true);
    store.setImportProgress({ source: "rekordbox", current: 50, total: 100 });
    const state = useLibraryStore.getState();
    expect(state.isImporting).toBe(true);
    expect(state.importProgress?.current).toBe(50);
  });
});

describe("useRecommendationStore", () => {
  beforeEach(() => {
    useRecommendationStore.setState({
      recommendations: [],
      isLoading: false,
      confidence: 0,
      error: null,
    });
  });

  it("should set recommendations with confidence", () => {
    const store = useRecommendationStore.getState();
    store.setRecommendations(
      [
        {
          trackId: "t1",
          score: 85,
          breakdown: { bpmScore: 90, keyScore: 80, energyScore: 85, genreScore: 50, historyScore: 50 },
          confidence: 75,
        },
      ],
      75,
    );
    const state = useRecommendationStore.getState();
    expect(state.recommendations).toHaveLength(1);
    expect(state.confidence).toBe(75);
  });

  it("should clear recommendations", () => {
    const store = useRecommendationStore.getState();
    store.setRecommendations(
      [{ trackId: "t1", score: 85, breakdown: { bpmScore: 90, keyScore: 80, energyScore: 85, genreScore: 50, historyScore: 50 }, confidence: 75 }],
      75,
    );
    store.clear();
    const state = useRecommendationStore.getState();
    expect(state.recommendations).toHaveLength(0);
    expect(state.confidence).toBe(0);
  });

  it("should handle errors", () => {
    const store = useRecommendationStore.getState();
    store.setError("Analysis failed");
    expect(useRecommendationStore.getState().error).toBe("Analysis failed");
  });
});

describe("usePlayerStore", () => {
  beforeEach(() => {
    usePlayerStore.setState({
      currentTrackId: null,
      title: null,
      artist: null,
      bpm: null,
      camelotKey: null,
      energy: null,
      durationMs: null,
      elapsedMs: 0,
      isPlaying: false,
      isPreviewPlaying: false,
      previewTrackId: null,
    });
  });

  it("should set current track", () => {
    const store = usePlayerStore.getState();
    store.setCurrentTrack({
      id: "t1",
      title: "Strobe",
      artist: "Deadmau5",
      bpm: 128,
      camelotKey: "8A",
      energy: 75,
      durationMs: 320000,
    });
    const state = usePlayerStore.getState();
    expect(state.currentTrackId).toBe("t1");
    expect(state.title).toBe("Strobe");
    expect(state.isPlaying).toBe(true);
  });

  it("should update elapsed time", () => {
    const store = usePlayerStore.getState();
    store.setElapsed(60000);
    expect(usePlayerStore.getState().elapsedMs).toBe(60000);
  });

  it("should manage preview playback", () => {
    const store = usePlayerStore.getState();
    store.startPreview("t2");
    expect(usePlayerStore.getState().isPreviewPlaying).toBe(true);
    expect(usePlayerStore.getState().previewTrackId).toBe("t2");

    store.stopPreview();
    expect(usePlayerStore.getState().isPreviewPlaying).toBe(false);
  });

  it("should clear track state", () => {
    const store = usePlayerStore.getState();
    store.setCurrentTrack({ id: "t1", title: "X", artist: "Y" });
    store.clearTrack();
    expect(usePlayerStore.getState().currentTrackId).toBeNull();
    expect(usePlayerStore.getState().isPlaying).toBe(false);
  });
});
