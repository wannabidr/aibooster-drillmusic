import { useCallback, useEffect, useState } from "react";
import { AppLayout } from "../layouts/AppLayout";
import { TrackListView } from "../components/TrackListView";
import { RecommendationPanel } from "../components/RecommendationPanel";
import { NowPlayingBar } from "../components/NowPlayingBar";
import { LibraryImportView } from "../components/LibraryImportView";
import { CamelotWheel } from "../components/CamelotWheel";
import { EnergyGraph } from "../components/EnergyGraph";
import { useLibraryStore } from "@infrastructure/state/useLibraryStore";
import { useRecommendationStore } from "@infrastructure/state/useRecommendationStore";
import { usePlayerStore } from "@infrastructure/state/usePlayerStore";
import { useThemeStore } from "@infrastructure/state/useThemeStore";
import { useRecommendations } from "../hooks/useRecommendations";

interface MainPageProps {
  onNavigateAnalytics?: () => void;
}

export function MainPage({ onNavigateAnalytics }: MainPageProps) {
  const library = useLibraryStore();
  const recs = useRecommendationStore();
  const player = usePlayerStore();
  const { performanceMode } = useThemeStore();
  const { fetchRecommendations } = useRecommendations();

  const [keyFilter, setKeyFilter] = useState<string | undefined>(undefined);
  const hasLibrary = library.tracks.length > 0;

  const handleTrackSelect = useCallback(
    (trackId: string) => {
      library.selectTrack(trackId);
      const track = library.tracks.find((t) => t.id === trackId);
      if (track) {
        player.setCurrentTrack({
          id: track.id,
          title: track.title,
          artist: track.artist,
          bpm: track.bpm,
          camelotKey: track.camelotKey,
          energy: track.energy,
          durationMs: track.durationMs,
        });
        fetchRecommendations(trackId);
      }
    },
    [library, player, fetchRecommendations],
  );

  const handleImport = useCallback(
    (_source: "rekordbox" | "serato" | "traktor" | "folder") => { // eslint-disable-line @typescript-eslint/no-unused-vars
      // Will be wired to Tauri IPC: importLibrary(source, path)
      library.setImporting(true);
    },
    [library],
  );

  // Refresh recommendations when selected track changes
  useEffect(() => {
    if (library.selectedTrackId) {
      fetchRecommendations(library.selectedTrackId);
    }
  }, [library.selectedTrackId, fetchRecommendations]);

  return (
    <AppLayout
      sidebar={
        onNavigateAnalytics ? (
          <nav>
            <button
              onClick={onNavigateAnalytics}
              data-testid="nav-analytics"
              style={{
                width: "100%",
                padding: "var(--space-sm) var(--space-md)",
                background: "none",
                border: "1px solid var(--border-default)",
                borderRadius: "var(--radius-md)",
                color: "var(--text-secondary)",
                cursor: "pointer",
                fontSize: "var(--font-size-sm)",
                textAlign: "left" as const,
              }}
            >
              Analytics
            </button>
          </nav>
        ) : undefined
      }
      main={
        hasLibrary ? (
          <TrackListView
            tracks={library.tracks.map((t) => ({
              id: t.id,
              title: t.title,
              artist: t.artist,
              bpm: t.bpm,
              key: t.camelotKey,
              energy: t.energy,
              durationMs: t.durationMs,
            }))}
            selectedTrackId={library.selectedTrackId ?? undefined}
            onTrackSelect={handleTrackSelect}
          />
        ) : (
          <LibraryImportView
            isImporting={library.isImporting}
            importProgress={library.importProgress ?? undefined}
            onImport={handleImport}
          />
        )
      }
      recommendations={
        hasLibrary ? (
          <>
            {player.camelotKey && (
              <CamelotWheel
                currentKey={player.camelotKey}
                selectedKey={keyFilter}
                onKeySelect={(key) =>
                  setKeyFilter(key === keyFilter ? undefined : key)
                }
              />
            )}
            {player.energy != null && (
              <EnergyGraph
                currentEnergyCurve={[player.energy]}
                direction="maintain"
              />
            )}
            <RecommendationPanel
              recommendations={recs.recommendations
                .filter(
                  (r) => {
                    if (!keyFilter) return true;
                    const track = library.tracks.find((t) => t.id === r.trackId);
                    return track?.camelotKey === keyFilter;
                  },
                )
                .map((r) => {
                  const track = library.tracks.find((t) => t.id === r.trackId);
                  return {
                    trackId: r.trackId,
                    title: track?.title ?? "Unknown",
                    artist: track?.artist ?? "Unknown",
                    score: r.score,
                    bpmScore: r.breakdown.bpmScore,
                    keyScore: r.breakdown.keyScore,
                    energyScore: r.breakdown.energyScore,
                    genreScore: r.breakdown.genreScore,
                    historyScore: r.breakdown.historyScore,
                    bpm: track?.bpm,
                    camelotPosition: track?.camelotKey,
                  };
                })}
              confidence={recs.confidence}
              isLoading={recs.isLoading}
              onSelect={handleTrackSelect}
              onPreview={(trackId) => player.startPreview(trackId)}
              glanceable
              performanceMode={performanceMode}
            />
          </>
        ) : undefined
      }
      nowPlaying={
        <NowPlayingBar
          title={player.title ?? undefined}
          artist={player.artist ?? undefined}
          bpm={player.bpm ?? undefined}
          camelotKey={player.camelotKey ?? undefined}
          energy={player.energy ?? undefined}
          elapsedMs={player.elapsedMs}
          durationMs={player.durationMs ?? undefined}
          glanceable
        />
      }
    />
  );
}
