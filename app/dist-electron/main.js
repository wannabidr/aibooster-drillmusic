import { app, BrowserWindow, ipcMain, screen } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { initDatabase, createSession } from './services/Database.js';
import { LogWatcher } from './services/LogWatcher.js';
import { MidiListener } from './services/MidiListener.js';
import { AIService } from './services/AIService.js';
import os from 'os';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
let mainWindow = null;
let logWatcher = null;
let midiListener = null;
let aiService = null;
let currentSessionId = null;
let currentTrack = null;
function createWindow() {
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.workAreaSize;
    mainWindow = new BrowserWindow({
        width: 300,
        height: 150,
        x: width - 320,
        y: height - 200,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
        },
        frame: false,
        transparent: true,
        alwaysOnTop: true,
        resizable: false,
        skipTaskbar: true,
    });
    if (process.env.VITE_DEV_SERVER_URL) {
        mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    }
    else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }
    return mainWindow;
}
async function initializeServices() {
    if (!mainWindow)
        return;
    try {
        // 1. Initialize Database
        console.log('Initializing database...');
        await initDatabase();
        currentSessionId = createSession();
        console.log(`Session created: ${currentSessionId}`);
        // 2. Initialize MIDI Listener
        console.log('Initializing MIDI listener...');
        midiListener = new MidiListener(mainWindow);
        const midiInputs = midiListener.listInputs();
        console.log(`Found ${midiInputs.length} MIDI input(s):`, midiInputs);
        // 3. Initialize Log Watcher
        console.log('Initializing log watcher...');
        logWatcher = new LogWatcher(mainWindow);
        // Set default log path (can be changed via IPC)
        const defaultLogPath = path.join(os.homedir(), 'drillmusic_test.log');
        console.log(`Default log path: ${defaultLogPath}`);
        // Don't start watching yet - wait for user to set path or use default
        // 4. Initialize AI Service
        console.log('Initializing AI service...');
        aiService = new AIService(mainWindow);
        // Determine Python script and index paths
        const isDev = process.env.VITE_DEV_SERVER_URL !== undefined;
        let pythonScriptPath;
        let indexDir;
        if (isDev) {
            // Development: go up from app/electron to project root
            const projectRoot = path.join(__dirname, '..', '..');
            pythonScriptPath = path.join(projectRoot, 'ai', 'src', 'main.py');
            indexDir = path.join(projectRoot, 'ai', 'src', 'index');
        }
        else {
            // Production: bundled resources
            const resourcesPath = process.resourcesPath;
            pythonScriptPath = path.join(resourcesPath, 'ai', 'main.py');
            indexDir = path.join(resourcesPath, 'ai', 'index');
        }
        console.log(`Python script: ${pythonScriptPath}`);
        console.log(`Index directory: ${indexDir}`);
        // Start AI service (async)
        aiService.start(pythonScriptPath, indexDir).catch((error) => {
            console.error('Failed to start AI service:', error);
            // Continue without AI service
        });
        console.log('All services initialized');
    }
    catch (error) {
        console.error('Error initializing services:', error);
    }
}
app.whenReady().then(async () => {
    createWindow();
    await initializeServices();
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
            initializeServices();
        }
    });
});
app.on('window-all-closed', () => {
    // Cleanup services
    if (logWatcher) {
        logWatcher.stopWatching();
    }
    if (midiListener) {
        midiListener.closeInput();
    }
    if (aiService) {
        aiService.stop();
    }
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
app.on('before-quit', () => {
    // Cleanup services
    if (logWatcher) {
        logWatcher.stopWatching();
    }
    if (midiListener) {
        midiListener.closeInput();
    }
    if (aiService) {
        aiService.stop();
    }
});
// ============================================
// IPC Handlers
// ============================================
// Get current track
ipcMain.handle('get-current-track', async () => {
    if (currentTrack) {
        return currentTrack;
    }
    return { title: 'Waiting for track...', artist: 'DrillMusic' };
});
// Get AI recommendations
ipcMain.handle('get-recommendations', async (_event, track) => {
    try {
        if (!aiService) {
            throw new Error('AI service not initialized');
        }
        if (!track || !track.filePath) {
            throw new Error('Track must have a filePath for AI recommendation');
        }
        const result = await aiService.getRecommendation({
            currentTrackPath: track.filePath,
            goal: track.goal || 'maintain',
            topK: 3
        });
        return result;
    }
    catch (error) {
        console.error('Error getting recommendations:', error);
        return {
            error: error.message,
            recommendations: []
        };
    }
});
// List MIDI devices
ipcMain.handle('list-midi-devices', async () => {
    try {
        if (!midiListener) {
            return [];
        }
        return midiListener.listInputs();
    }
    catch (error) {
        console.error('Error listing MIDI devices:', error);
        return [];
    }
});
// Select MIDI device
ipcMain.handle('select-midi-device', async (_event, deviceId) => {
    try {
        if (!midiListener) {
            throw new Error('MIDI listener not initialized');
        }
        midiListener.closeInput();
        midiListener.openInput(deviceId);
        return { success: true };
    }
    catch (error) {
        console.error('Error selecting MIDI device:', error);
        return { success: false, error: error.message };
    }
});
// Set log file path
ipcMain.handle('set-log-path', async (_event, logPath) => {
    try {
        if (!logWatcher) {
            throw new Error('Log watcher not initialized');
        }
        logWatcher.stopWatching();
        logWatcher.startWatching(logPath);
        return { success: true };
    }
    catch (error) {
        console.error('Error setting log path:', error);
        return { success: false, error: error.message };
    }
});
// Manual track update (for testing)
ipcMain.handle('set-current-track', async (_event, track) => {
    try {
        currentTrack = track;
        if (mainWindow) {
            mainWindow.webContents.send('track-update', track);
        }
        return { success: true };
    }
    catch (error) {
        console.error('Error setting current track:', error);
        return { success: false, error: error.message };
    }
});
