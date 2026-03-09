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
import { importLibrary, getTracks } from "@infrastructure/tauri-bridge/invoke";

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
    async (source: "rekordbox" | "serato" | "traktor" | "folder") => {
      console.log(`[MainPage] handleImport called with source: ${source}`);

      const isTauri = typeof window !== "undefined" && "__TAURI__" in window;
      console.log(`[MainPage] Tauri available: ${isTauri}`);

      // Open file/folder picker via Tauri dialog
      let selectedPath: string | null = null;
      if (isTauri) {
        try {
          // eslint-disable-next-line @typescript-eslint/ban-ts-comment
          // @ts-ignore - @tauri-apps/plugin-dialog is provided at runtime by Tauri
          const { open } = await import("@tauri-apps/plugin-dialog");
          if (source === "folder") {
            selectedPath = await open({ directory: true, title: "Select music folder" }) as string | null;
          } else {
            const filters: Record<string, { name: string; extensions: string[] }> = {
              rekordbox: { name: "Rekordbox XML", extensions: ["xml"] },
              serato: { name: "Serato Database", extensions: ["*"] },
              traktor: { name: "Traktor NML", extensions: ["nml"] },
            };
            selectedPath = await open({
              title: `Select ${source} library file`,
              filters: [filters[source]],
            }) as string | null;
          }
        } catch (err) {
          console.error("[MainPage] File dialog error:", err);
          console.error("[MainPage] @tauri-apps/plugin-dialog may not be installed. Run: npm install @tauri-apps/plugin-dialog");
          alert(`File dialog failed: ${err}. Check console for details.`);
          return;
        }
      } else {
        console.warn("[MainPage] Not running in Tauri. Import requires Tauri desktop environment.");
        console.warn("[MainPage] Run 'npx tauri dev' instead of 'npm run dev'");
        alert("Import requires Tauri environment. Run 'npx tauri dev' instead of 'npm run dev'.");
        return;
      }

      if (!selectedPath) {
        console.log("[MainPage] File dialog cancelled by user");
        return;
      }

      console.log(`[MainPage] Selected path: ${selectedPath}`);
      library.setImporting(true);

      try {
        const result = await importLibrary(source, selectedPath);
        console.log("[MainPage] Import result:", result);

        if (result.errors.length > 0) {
          console.warn("[MainPage] Import had errors:", result.errors);
        }

        // Fetch tracks after import
        console.log("[MainPage] Fetching tracks after import...");
        const tracks = await getTracks();
        console.log(`[MainPage] Loaded ${tracks.length} tracks`);
        library.setTracks(
          tracks.map((t) => ({
            id: t.id,
            title: t.title,
            artist: t.artist,
            album: t.album,
            filePath: t.filePath,
            durationMs: t.durationMs,
            bpm: t.bpm,
            camelotKey: t.camelotPosition,
            energy: t.energy,
            genre: t.genre,
            analyzed: !!t.analyzedAt,
          })),
        );
      } catch (err) {
        console.error("[MainPage] Import failed:", err);
        alert(`Import failed: ${err}`);
      } finally {
        library.setImporting(false);
        library.setImportProgress(null);
        console.log("[MainPage] Import flow complete");
      }
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
