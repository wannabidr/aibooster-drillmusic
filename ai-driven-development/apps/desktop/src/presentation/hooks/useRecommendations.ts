import { useCallback } from "react";
import { useLibraryStore } from "@infrastructure/state/useLibraryStore";
import { useRecommendationStore } from "@infrastructure/state/useRecommendationStore";
import { GetRecommendations } from "@application/use-cases/GetRecommendations";
import { InMemoryAnalysisDataProvider } from "@infrastructure/repositories/InMemoryAnalysisDataProvider";

export function useRecommendations() {
  const tracks = useLibraryStore((s) => s.tracks);
  const { setRecommendations, setLoading, setError } = useRecommendationStore();

  const fetchRecommendations = useCallback(
    (currentTrackId: string, recentTrackIds: string[] = []) => {
      setLoading(true);

      try {
        // Build analysis data from library tracks
        const provider = new InMemoryAnalysisDataProvider();
        for (const track of tracks) {
          if (track.analyzed && track.bpm && track.camelotKey && track.energy != null) {
            provider.add({
              trackId: track.id,
              bpm: track.bpm,
              camelotPosition: track.camelotKey,
              energy: track.energy,
              genre: track.genre,
            });
          }
        }

        const useCase = new GetRecommendations(provider);
        const results = useCase.execute({
          currentTrackId,
          limit: 10,
          excludeTrackIds: recentTrackIds,
        });

        const confidence = results.length > 0 ? (results[0]?.confidence ?? 0) : 0;
        setRecommendations(results, confidence);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to get recommendations");
      }
    },
    [tracks, setRecommendations, setLoading, setError],
  );

  return { fetchRecommendations };
}
