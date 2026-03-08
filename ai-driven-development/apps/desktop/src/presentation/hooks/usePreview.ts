import { useCallback, useRef, useState } from "react";
import { GeneratePreview, PreviewResult } from "@application/use-cases/GeneratePreview";
import { usePlayerStore } from "@infrastructure/state/usePlayerStore";

export function usePreview() {
  const generatorRef = useRef(new GeneratePreview());
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { startPreview, stopPreview } = usePlayerStore();

  const loadPreview = useCallback(
    async (fromTrackId: string, toTrackId: string) => {
      setIsLoading(true);
      setError(null);
      startPreview(toTrackId);

      try {
        const result = await generatorRef.current.execute({
          fromTrackId,
          toTrackId,
        });
        setPreview(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Preview generation failed");
        stopPreview();
      } finally {
        setIsLoading(false);
      }
    },
    [startPreview, stopPreview],
  );

  const closePreview = useCallback(() => {
    setPreview(null);
    stopPreview();
  }, [stopPreview]);

  const preRenderTop = useCallback(
    async (currentTrackId: string, recommendedIds: string[]) => {
      try {
        await generatorRef.current.preRenderTopN(currentTrackId, recommendedIds, 3);
      } catch {
        // Pre-rendering failures are non-critical
      }
    },
    [],
  );

  return {
    preview,
    isLoading,
    error,
    loadPreview,
    closePreview,
    preRenderTop,
  };
}
