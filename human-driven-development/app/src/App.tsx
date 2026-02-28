import React, { useEffect } from 'react';
import { useAppStore } from './stores/useAppStore';
import { Music, Disc3, Loader2, AlertCircle } from 'lucide-react';

function App() {
  const { 
    currentTrack, 
    recommendations, 
    isLoadingRecommendations,
    setCurrentTrack, 
    setRecommendations,
    setLoadingRecommendations
  } = useAppStore();

  useEffect(() => {
    if (!window.electronAPI) return;

    // Listen for track updates
    const unsubscribeTrack = window.electronAPI.onTrackUpdate((_event, track) => {
      console.log('Track update received:', track);
      setCurrentTrack(track);
    });

    // Listen for recommendation updates
    const unsubscribeRecs = window.electronAPI.onRecommendationsUpdate((_event, response) => {
      console.log('Recommendations update received:', response);
      setLoadingRecommendations(false);
      if (response.error) {
        console.error('Recommendation error:', response.error);
        setRecommendations([]);
      } else {
        setRecommendations(response.recommendations || []);
      }
    });

    // Initial fetch
    window.electronAPI.getCurrentTrack().then((track) => {
      console.log('Initial track:', track);
      setCurrentTrack(track);
    });

    return () => {
      unsubscribeTrack();
      unsubscribeRecs();
    };
  }, [setCurrentTrack, setRecommendations, setLoadingRecommendations]);

  // Request recommendations when track changes
  useEffect(() => {
    if (currentTrack && currentTrack.filePath && window.electronAPI) {
      console.log('Requesting recommendations for:', currentTrack.title);
      setLoadingRecommendations(true);
      setRecommendations([]);
      
      window.electronAPI.getRecommendations(currentTrack)
        .then((response) => {
          console.log('Recommendations response:', response);
          setLoadingRecommendations(false);
          if (response.error) {
            console.error('Recommendation error:', response.error);
          } else {
            setRecommendations(response.recommendations || []);
          }
        })
        .catch((error) => {
          console.error('Failed to get recommendations:', error);
          setLoadingRecommendations(false);
        });
    }
  }, [currentTrack?.filePath]);

  const formatTrackName = (rec: any) => {
    if (rec.title && rec.artist) {
      return `${rec.title} - ${rec.artist}`;
    } else if (rec.title) {
      return rec.title;
    } else if (rec.path) {
      const filename = rec.path.split('/').pop()?.replace(/\.[^/.]+$/, '') || 'Unknown';
      return filename;
    }
    return 'Unknown Track';
  };

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
        {currentTrack?.bpm && (
          <div className="text-xs text-gray-500 mt-1">
            {currentTrack.bpm} BPM {currentTrack.key && `• ${currentTrack.key}`}
          </div>
        )}
      </div>

      {/* Recommendations Preview */}
      <div className="mt-3 pt-3 border-t border-gray-700/50">
        <div className="flex items-center gap-2 text-xs text-gray-400 mb-2">
          <Music size={12} />
          <span>NEXT UP</span>
          {isLoadingRecommendations && (
            <Loader2 size={10} className="animate-spin ml-1" />
          )}
        </div>
        
        <div className="space-y-1">
          {isLoadingRecommendations ? (
            <div className="text-xs text-gray-500 italic">Loading recommendations...</div>
          ) : recommendations.length > 0 ? (
            recommendations.slice(0, 3).map((rec, idx) => (
              <div 
                key={rec.track_id || idx} 
                className="text-xs truncate text-gray-300 hover:text-white cursor-pointer transition-colors flex items-center justify-between gap-2"
                title={formatTrackName(rec)}
              >
                <span className="truncate">
                  {idx + 1}. {formatTrackName(rec)}
                </span>
                {rec.bpm && (
                  <span className="text-gray-500 text-[10px] flex-shrink-0">
                    {Math.round(rec.bpm)}
                  </span>
                )}
              </div>
            ))
          ) : currentTrack?.filePath ? (
            <div className="text-xs text-gray-500 flex items-center gap-1">
              <AlertCircle size={10} />
              <span>No recommendations available</span>
            </div>
          ) : (
            <div className="text-xs text-gray-500 italic">
              Set a track with filePath to get recommendations
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
