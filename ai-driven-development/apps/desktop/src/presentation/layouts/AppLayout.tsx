import { type ReactNode } from "react";

interface AppLayoutProps {
  sidebar?: ReactNode;
  main: ReactNode;
  nowPlaying?: ReactNode;
  recommendations?: ReactNode;
}

export function AppLayout({ sidebar, main, nowPlaying, recommendations }: AppLayoutProps) {
  return (
    <div style={styles.container}>
      {sidebar && <aside style={styles.sidebar}>{sidebar}</aside>}
      <div style={styles.content}>
        <main style={styles.main}>{main}</main>
        {recommendations && (
          <aside style={styles.recommendations}>{recommendations}</aside>
        )}
      </div>
      {nowPlaying && <footer style={styles.nowPlaying}>{nowPlaying}</footer>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    backgroundColor: "var(--bg-base)",
    color: "var(--text-primary)",
  },
  sidebar: {
    width: 240,
    backgroundColor: "var(--bg-surface)",
    borderRight: "1px solid var(--border-subtle)",
    padding: "var(--space-md)",
    overflowY: "auto",
    flexShrink: 0,
  },
  content: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
  },
  main: {
    flex: 1,
    overflow: "auto",
    padding: "var(--space-md)",
  },
  recommendations: {
    width: 320,
    backgroundColor: "var(--bg-surface)",
    borderLeft: "1px solid var(--border-subtle)",
    padding: "var(--space-md)",
    overflowY: "auto",
    flexShrink: 0,
  },
  nowPlaying: {
    height: 80,
    backgroundColor: "var(--bg-elevated)",
    borderTop: "1px solid var(--border-subtle)",
    padding: "var(--space-sm) var(--space-md)",
    display: "flex",
    alignItems: "center",
    flexShrink: 0,
  },
};
