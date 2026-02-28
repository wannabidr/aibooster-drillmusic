/// <reference types="vite/client" />

interface Track {
  title: string;
  artist: string;
  bpm?: number;
  key?: string;
  filePath?: string;
  goal?: 'maintain' | 'up' | 'down' | 'peak';
}

interface Recommendation {
  track_id: string;
  path: string;
  title?: string;
  artist?: string;
  score: number;
  bpm?: number;
  key?: string;
}

interface RecommendationResponse {
  recommendations: Recommendation[];
  error?: string;
}

interface MidiDevice {
  id: number;
  name: string;
}

interface MidiMessage {
  timestamp: number;
  status: number;
  control: number;
  value: number;
}

interface Window {
  electronAPI: {
    // Track management
    getCurrentTrack: () => Promise<Track>;
    setCurrentTrack: (track: Track) => Promise<{ success: boolean; error?: string }>;
    onTrackUpdate: (callback: (event: any, track: Track) => void) => () => void;

    // AI recommendations
    getRecommendations: (track: Track) => Promise<RecommendationResponse>;
    onRecommendationsUpdate: (callback: (event: any, response: RecommendationResponse) => void) => () => void;

    // MIDI devices
    listMidiDevices: () => Promise<MidiDevice[]>;
    selectMidiDevice: (deviceId: number) => Promise<{ success: boolean; error?: string }>;
    onMidiMessage: (callback: (event: any, message: MidiMessage) => void) => () => void;

    // Log file watching
    setLogPath: (path: string) => Promise<{ success: boolean; error?: string }>;
  };
}
