# Analysis Package

Python audio analysis engine and ML pipeline for AI DJ Assist.

## Features

- **Audio Analysis**: BPM detection, key detection, energy analysis (via Essentia)
- **Audio Fingerprinting**: Chromaprint-based track identification
- **ML Recommendation**: PyTorch MLP model with ONNX export for on-device inference
- **Genre Classification**: 42-dim feature extraction, 64-dim embeddings, cosine similarity
- **Library Import**: Rekordbox (XML), Traktor (NML), Serato (binary), Mixcloud, SoundCloud, CSV
- **JSON-RPC Server**: Sidecar communication with the Tauri desktop app

## Architecture

Clean Architecture with strict layer separation:

- `domain/` - Entities, value objects, repository ports (no external deps)
- `application/` - Use cases (AnalyzeTrack, BatchAnalyze, ImportLibrary)
- `infrastructure/` - Essentia, Chromaprint, SQLite, ONNX adapters
- `interface/` - JSON-RPC server and handlers

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,audio,ml]"
```

## Test

```bash
pytest
```

## Code Style

```bash
ruff check .
black --check .
```
