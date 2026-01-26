import React, { useEffect } from 'react';
import { useAppStore } from './stores/useAppStore';
import { Music, Disc3 } from 'lucide-react';

function App() {
  const { currentTrack, setCurrentTrack } = useAppStore();

  useEffect(() => {
    // Listen for track updates from Electron main process
    if (window.electronAPI) {
      window.electronAPI.onTrackUpdate((_event: any, track: any) => {
        setCurrentTrack(track);
      });
      
      // Initial fetch
      window.electronAPI.getCurrentTrack().then((track: any) => {
        setCurrentTrack(track);
      });
    }
  }, [setCurrentTrack]);

  return (
    <div className="w-full h-screen bg-black/80 text-white p-4 rounded-xl border border-gray-700 shadow-2xl backdrop-blur-md overflow-hidden flex flex-col select-none draggable">
      {/* Header / Drag Handle */}
      <div className="flex items-center justify-between mb-2 opacity-50 hover:opacity-100 transition-opacity">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-green-400">
          <Disc3 size={14} className="animate-spin-slow" />
          <span>DrillMusic AI</span>
        </div>
        <div className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.8)]"></div>
      </div>

      {/* Current Track Info */}
      <div className="flex-1 flex flex-col justify-center">
        <div className="text-xs text-gray-400 mb-1">NOW PLAYING</div>
        <h1 className="text-lg font-bold leading-tight truncate">
          {currentTrack?.title || 'Waiting for track...'}
        </h1>
        <p className="text-sm text-gray-300 truncate">
          {currentTrack?.artist || 'DrillMusic'}
        </p>
      </div>

      {/* Recommendations Preview (Mini) */}
      <div className="mt-3 pt-3 border-t border-gray-700/50">
        <div className="flex items-center gap-2 text-xs text-gray-400 mb-2">
          <Music size={12} />
          <span>NEXT UP</span>
        </div>
        <div className="space-y-1">
          <div className="text-xs truncate text-gray-300 hover:text-white cursor-pointer transition-colors">
            1. Example Rec 1 - Artist
          </div>
          <div className="text-xs truncate text-gray-500 hover:text-white cursor-pointer transition-colors">
            2. Example Rec 2 - Artist
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
