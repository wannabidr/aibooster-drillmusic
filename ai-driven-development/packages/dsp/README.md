# DSP Package

Rust DSP engine for audio processing, MIDI control, and virtual audio routing.

## Features

- **AI Audio Blend**: Crossfade with EQ automation, filter sweeps, phrase detection, 5 blend styles
- **MIDI Output**: Pioneer, Denon, Native Instruments protocols, SysEx, 14-bit BPM
- **Virtual Audio Routing**: Cross-platform output via cpal
- **Serato Parser**: Binary crate format reader

## Build

```bash
rustup run stable cargo build
```

## Test

```bash
rustup run stable cargo test
```

## Code Style

```bash
rustup run stable cargo clippy
rustup run stable cargo fmt --check
```
