import Database from 'better-sqlite3';
import path from 'path';
import { app } from 'electron';

const dbPath = path.join(app.getPath('userData'), 'drillmusic.db');
const db = new Database(dbPath);

export function initDatabase() {
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
      event_type TEXT, -- 'play', 'pause', 'cue', 'transition'
      midi_snapshot TEXT, -- JSON string of EQ/Fader values
      FOREIGN KEY(session_id) REFERENCES sessions(id),
      FOREIGN KEY(track_id) REFERENCES tracks(id)
    );
  `);
}

export function createSession() {
  const stmt = db.prepare('INSERT INTO sessions (start_time) VALUES (CURRENT_TIMESTAMP)');
  return stmt.run().lastInsertRowid;
}

export function getTrackByPath(filePath: string) {
  const stmt = db.prepare('SELECT * FROM tracks WHERE file_path = ?');
  return stmt.get(filePath);
}

export function insertTrack(track: { title: string; artist: string; filePath: string }) {
  const stmt = db.prepare('INSERT INTO tracks (title, artist, file_path) VALUES (@title, @artist, @filePath)');
  return stmt.run(track).lastInsertRowid;
}

export function logMixEvent(sessionId: number | bigint, trackId: number | bigint, eventType: string, midiSnapshot: any) {
  const stmt = db.prepare('INSERT INTO mix_logs (session_id, track_id, event_type, midi_snapshot) VALUES (@sessionId, @trackId, @eventType, @midiSnapshot)');
  stmt.run({
    sessionId,
    trackId,
    eventType,
    midiSnapshot: JSON.stringify(midiSnapshot),
  });
}

export default db;
