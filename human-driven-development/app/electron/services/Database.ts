import Database from 'better-sqlite3';
import path from 'path';

let db: Database.Database | null = null;

// Lazy load electron app for production use only
let electronApp: Electron.App | null = null;
async function getElectronApp(): Promise<Electron.App> {
  if (!electronApp) {
    const { app } = await import('electron');
    electronApp = app;
  }
  return electronApp;
}

// Initialize database with optional custom path (for testing)
export async function initDatabase(customPath?: string) {
  try {
    let dbPath: string;
    
    if (customPath) {
      dbPath = customPath;
    } else {
      const app = await getElectronApp();
      dbPath = path.join(app.getPath('userData'), 'drillmusic.db');
    }
    
    db = new Database(dbPath);
    
    // Enable foreign keys
    db.pragma('foreign_keys = ON');
    
    db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        end_time DATETIME
      );

      CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        artist TEXT,
        bpm REAL,
        key TEXT,
        energy REAL,
        file_path TEXT UNIQUE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );

      CREATE TABLE IF NOT EXISTS mix_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        track_id INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT,
        midi_snapshot TEXT,
        FOREIGN KEY(session_id) REFERENCES sessions(id),
        FOREIGN KEY(track_id) REFERENCES tracks(id)
      );
    `);
    
    return db;
  } catch (error) {
    console.error('Failed to initialize database:', error);
    throw error;
  }
}

function getDb(): Database.Database {
  if (!db) {
    throw new Error('Database not initialized. Call initDatabase() first.');
  }
  return db;
}

export function createSession() {
  const database = getDb();
  const stmt = database.prepare('INSERT INTO sessions (start_time) VALUES (CURRENT_TIMESTAMP)');
  return stmt.run().lastInsertRowid;
}

export function getTrackByPath(filePath: string) {
  const database = getDb();
  const stmt = database.prepare('SELECT * FROM tracks WHERE file_path = ?');
  return stmt.get(filePath);
}

export function insertTrack(track: { title: string; artist: string; filePath: string }) {
  const database = getDb();
  const stmt = database.prepare('INSERT INTO tracks (title, artist, file_path) VALUES (@title, @artist, @filePath)');
  return stmt.run(track).lastInsertRowid;
}

export function logMixEvent(sessionId: number | bigint, trackId: number | bigint, eventType: string, midiSnapshot: any) {
  const database = getDb();
  const stmt = database.prepare('INSERT INTO mix_logs (session_id, track_id, event_type, midi_snapshot) VALUES (@sessionId, @trackId, @eventType, @midiSnapshot)');
  stmt.run({
    sessionId,
    trackId,
    eventType,
    midiSnapshot: JSON.stringify(midiSnapshot),
  });
}

export function closeDatabase() {
  if (db) {
    db.close();
    db = null;
  }
}

export function getDatabase() {
  return db;
}
