# AI DJ Assist

An intelligent desktop companion for professional club and festival DJs. AI DJ Assist runs alongside existing DJ software (Rekordbox, Serato, Traktor) to deliver real-time track recommendations, harmonic mixing suggestions, and AI-rendered audio blend previews -- reducing cognitive load during performance while respecting creative control.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Desktop App | Tauri v2 + React + TypeScript |
| Audio Analysis & ML | Python 3.11+ (Essentia, PyTorch, ONNX) |
| DSP & Audio | Rust (cpal, midir) |
| Community Backend | FastAPI + PostgreSQL |
| Billing | Stripe (Pro Monthly / Annual) |

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.11+
- Rust (stable toolchain)
- PostgreSQL (for backend only)

### Setup

```bash
# Install Node dependencies
npm install

# Set up Python analysis package
cd packages/analysis
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,audio,ml]"

# Build Rust DSP
cd packages/dsp
rustup run stable cargo build

# Copy environment template
cp .env.example .env
# Edit .env with your values

# Run the desktop app
npm run dev
```

## Project Structure

```
ai-driven-development/
├── apps/
│   └── desktop/          # Tauri + React desktop app
├── packages/
│   ├── analysis/         # Python: audio analysis, ML, library parsers
│   ├── backend/          # FastAPI community backend
│   └── dsp/              # Rust DSP engine, MIDI, virtual audio
├── shared/
│   └── types/            # Shared TypeScript type definitions
├── scripts/              # Build & architecture check scripts
└── docs/
    └── plans/            # Implementation plan & architecture docs
```

## Architecture

Clean Architecture with strict 4-layer separation:

```
Presentation -> Application -> Domain <- Infrastructure
```

- **Domain**: Entities, value objects, ports (zero external dependencies)
- **Application**: Use cases and orchestration
- **Infrastructure**: DB, parsers, APIs, audio engines, ML runtime
- **Presentation**: UI components, Tauri IPC, state management

Full architecture details: [docs/plans/PLAN_AI_DJ_ASSIST.md](docs/plans/PLAN_AI_DJ_ASSIST.md)

## Testing

```bash
# Run all desktop tests
npm test

# Python analysis tests
cd packages/analysis && pytest

# Python backend tests
cd packages/backend && pytest

# Rust DSP tests
cd packages/dsp && rustup run stable cargo test

# Architecture dependency guard
npm run check:arch
```

## License

TBD
