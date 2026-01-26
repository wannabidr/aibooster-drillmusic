/// <reference types="vite/client" />

interface Window {
  electronAPI: {
    getCurrentTrack: () => Promise<any>;
    onTrackUpdate: (callback: (event: any, value: any) => void) => void;
  };
}
