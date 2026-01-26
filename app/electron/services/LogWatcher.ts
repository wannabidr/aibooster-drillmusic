import chokidar from 'chokidar';
import fs from 'fs';
import { BrowserWindow } from 'electron';

// This is a placeholder for Rekordbox/Serato history file parsing.
// Actual implementation depends on the specific format of the history file.
// For now, we'll simulate watching a dummy log file.

export class LogWatcher {
  private watcher: chokidar.FSWatcher | null = null;
  private mainWindow: BrowserWindow;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;
  }

  startWatching(logPath: string) {
    if (this.watcher) {
      this.watcher.close();
    }

    this.watcher = chokidar.watch(logPath, {
      persistent: true,
      usePolling: true, // Often necessary for log files
    });

    this.watcher.on('change', (path) => {
      this.handleFileChange(path);
    });

    console.log(`Started watching: ${logPath}`);
  }

  private handleFileChange(filePath: string) {
    // In a real scenario, we would read the last few lines of the file
    // and parse them to find the currently playing track.
    fs.readFile(filePath, 'utf-8', (err, data) => {
      if (err) {
        console.error('Error reading log file:', err);
        return;
      }
      
      // Dummy parsing logic
      const lines = data.trim().split('\n');
      const lastLine = lines[lines.length - 1];
      
      if (lastLine) {
        // Assume format: "TIMESTAMP | ARTIST - TITLE"
        const parts = lastLine.split('|');
        if (parts.length >= 2) {
          const trackInfo = parts[1].trim();
          const [artist, title] = trackInfo.split('-').map(s => s.trim());
          
          this.mainWindow.webContents.send('track-update', {
            artist: artist || 'Unknown Artist',
            title: title || 'Unknown Track',
          });
        }
      }
    });
  }

  stopWatching() {
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }
  }
}
