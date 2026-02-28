import { contextBridge, ipcRenderer, IpcRendererEvent } from 'electron';

export interface Track {
  title: string;
  artist: string;
  bpm?: number;
  key?: string;
  filePath?: string;
  goal?: 'maintain' | 'up' | 'down' | 'peak';
}

export interface Recommendation {
  track_id: string;
  path: string;
  title?: string;
  artist?: string;
  score: number;
  bpm?: number;
  key?: string;
}

export interface RecommendationResponse {
  recommendations: Recommendation[];
  error?: string;
}

export interface MidiDevice {
  id: number;
  name: string;
}

export interface MidiMessage {
  timestamp: number;
  status: number;
  control: number;
  value: number;
}

contextBridge.exposeInMainWorld('electronAPI', {
  // Track management
  getCurrentTrack: () => ipcRenderer.invoke('get-current-track'),
  setCurrentTrack: (track: Track) => ipcRenderer.invoke('set-current-track', track),
  onTrackUpdate: (callback: (event: IpcRendererEvent, track: Track) => void) => {
    ipcRenderer.on('track-update', callback);
    return () => ipcRenderer.removeListener('track-update', callback);
  },

  // AI recommendations
  getRecommendations: (track: Track) => ipcRenderer.invoke('get-recommendations', track),
  onRecommendationsUpdate: (callback: (event: IpcRendererEvent, response: RecommendationResponse) => void) => {
    ipcRenderer.on('recommendations-update', callback);
    return () => ipcRenderer.removeListener('recommendations-update', callback);
  },

  // MIDI devices
  listMidiDevices: () => ipcRenderer.invoke('list-midi-devices'),
  selectMidiDevice: (deviceId: number) => ipcRenderer.invoke('select-midi-device', deviceId),
  onMidiMessage: (callback: (event: IpcRendererEvent, message: MidiMessage) => void) => {
    ipcRenderer.on('midi-message', callback);
    return () => ipcRenderer.removeListener('midi-message', callback);
  },

  // Log file watching
  setLogPath: (path: string) => ipcRenderer.invoke('set-log-path', path),
});
