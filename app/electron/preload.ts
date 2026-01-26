import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  getCurrentTrack: () => ipcRenderer.invoke('get-current-track'),
  onTrackUpdate: (callback: (event: any, value: any) => void) => ipcRenderer.on('track-update', callback),
});
