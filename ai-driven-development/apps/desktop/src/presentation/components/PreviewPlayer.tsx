import { useCallback, useEffect, useRef, useState } from "react";

interface PreviewPlayerProps {
  audioUrl?: string;
  durationMs?: number;
  transitionPointMs?: number;
  isLoading: boolean;
  onClose: () => void;
}

export function PreviewPlayer({
  audioUrl,
  durationMs,
  transitionPointMs,
  isLoading,
  onClose,
}: PreviewPlayerProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentMs, setCurrentMs] = useState(0);

  useEffect(() => {
    if (audioUrl) {
      const audio = new Audio(audioUrl);
      audioRef.current = audio;

      audio.addEventListener("timeupdate", () => {
        setCurrentMs(audio.currentTime * 1000);
      });
      audio.addEventListener("ended", () => {
        setIsPlaying(false);
        setCurrentMs(0);
      });

      return () => {
        audio.pause();
        audio.src = "";
      };
    }
  }, [audioUrl]);

  const togglePlayback = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(() => {});
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  const progress = durationMs ? (currentMs / durationMs) * 100 : 0;
  const transitionProgress = durationMs && transitionPointMs
    ? (transitionPointMs / durationMs) * 100
    : 50;

  if (isLoading) {
    return (
      <div style={styles.container}>
        <div style={styles.loading}>
          <div style={styles.spinner} />
          <span>Rendering preview...</span>
        </div>
      </div>
    );
  }

  if (!audioUrl) return null;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>Blend Preview</span>
        <button style={styles.closeBtn} onClick={onClose}>
          X
        </button>
      </div>

      <div style={styles.waveformContainer}>
        {/* Progress bar with transition marker */}
        <div style={styles.progressTrack}>
          <div style={{ ...styles.progressBar, width: `${progress}%` }} />
          <div
            style={{
              ...styles.transitionMarker,
              left: `${transitionProgress}%`,
            }}
          />
        </div>
        <div style={styles.labels}>
          <span>Track A</span>
          <span style={styles.transitionLabel}>Transition</span>
          <span>Track B</span>
        </div>
      </div>

      <div style={styles.controls}>
        <button style={styles.playBtn} onClick={togglePlayback}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <span style={styles.time}>
          {formatTime(currentMs)} / {formatTime(durationMs ?? 0)}
        </span>
      </div>
    </div>
  );
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: "var(--bg-elevated)",
    borderRadius: "var(--radius-md)",
    border: "1px solid var(--border-default)",
    padding: "var(--space-md)",
    marginTop: "var(--space-md)",
  },
  loading: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "var(--space-sm)",
    color: "var(--text-muted)",
    padding: "var(--space-lg)",
    fontSize: "var(--font-size-sm)",
  },
  spinner: {
    width: 16,
    height: 16,
    border: "2px solid var(--border-default)",
    borderTopColor: "var(--accent-primary)",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "var(--space-sm)",
  },
  title: {
    fontSize: "var(--font-size-sm)",
    fontWeight: 600,
  },
  closeBtn: {
    background: "none",
    border: "none",
    color: "var(--text-muted)",
    cursor: "pointer",
    fontSize: "var(--font-size-sm)",
    padding: "var(--space-xs)",
  },
  waveformContainer: {
    marginBottom: "var(--space-md)",
  },
  progressTrack: {
    position: "relative",
    height: 32,
    backgroundColor: "var(--bg-overlay)",
    borderRadius: "var(--radius-sm)",
    overflow: "hidden",
  },
  progressBar: {
    position: "absolute",
    top: 0,
    left: 0,
    height: "100%",
    backgroundColor: "var(--accent-primary)",
    opacity: 0.3,
    transition: "width 100ms linear",
  },
  transitionMarker: {
    position: "absolute",
    top: 0,
    width: 2,
    height: "100%",
    backgroundColor: "var(--accent-warm)",
    transform: "translateX(-50%)",
  },
  labels: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: "var(--space-xs)",
    fontSize: "var(--font-size-xs)",
    color: "var(--text-muted)",
  },
  transitionLabel: {
    color: "var(--accent-warm)",
  },
  controls: {
    display: "flex",
    alignItems: "center",
    gap: "var(--space-md)",
  },
  playBtn: {
    padding: "var(--space-xs) var(--space-lg)",
    backgroundColor: "var(--accent-primary)",
    color: "var(--text-primary)",
    border: "none",
    borderRadius: "var(--radius-sm)",
    cursor: "pointer",
    fontSize: "var(--font-size-sm)",
    fontWeight: 500,
  },
  time: {
    fontSize: "var(--font-size-xs)",
    fontFamily: "var(--font-mono)",
    color: "var(--text-muted)",
  },
};
