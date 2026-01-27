import { contextBridge, ipcRenderer } from 'electron';
contextBridge.exposeInMainWorld('electronAPI', {
    // Track management
    getCurrentTrack: () => ipcRenderer.invoke('get-current-track'),
    setCurrentTrack: (track) => ipcRenderer.invoke('set-current-track', track),
    onTrackUpdate: (callback) => {
        ipcRenderer.on('track-update', callback);
        return () => ipcRenderer.removeListener('track-update', callback);
    },
    // AI recommendations
    getRecommendations: (track) => ipcRenderer.invoke('get-recommendations', track),
    onRecommendationsUpdate: (callback) => {
        ipcRenderer.on('recommendations-update', callback);
        return () => ipcRenderer.removeListener('recommendations-update', callback);
    },
    // MIDI devices
    listMidiDevices: () => ipcRenderer.invoke('list-midi-devices'),
    selectMidiDevice: (deviceId) => ipcRenderer.invoke('select-midi-device', deviceId),
    onMidiMessage: (callback) => {
        ipcRenderer.on('midi-message', callback);
        return () => ipcRenderer.removeListener('midi-message', callback);
    },
    // Log file watching
    setLogPath: (path) => ipcRenderer.invoke('set-log-path', path),
});
