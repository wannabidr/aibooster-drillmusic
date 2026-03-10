#!/usr/bin/env python3
"""
AI DJ Assist - Interactive CLI for testing analysis & recommendation.

Usage:
    source .venv/bin/activate
    python cli.py /path/to/music/folder

Commands (after scan):
    list              - Show all analyzed tracks
    select <number>   - Select a track as "now playing"
    recommend         - Get next track recommendations for selected track
    detail <number>   - Show detailed analysis for a track
    rescan            - Re-scan the music directory
    quit              - Exit
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.domain.entities.analysis_result import AnalysisResult
from src.domain.value_objects.key_signature import KeySignature

SUPPORTED_EXTENSIONS = {".wav", ".aiff", ".aif", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Compatibility scoring (pure Python, no ML needed) ──────────────────


def bpm_compatibility(bpm_a: float, bpm_b: float) -> float:
    """Score 0-1 based on BPM closeness. Handles double/half time."""
    if bpm_a == 0 or bpm_b == 0:
        return 0.0
    ratio = bpm_a / bpm_b
    # Check direct, double, half time
    best_diff = min(
        abs(ratio - 1.0),
        abs(ratio - 2.0),
        abs(ratio - 0.5),
    )
    # Within 6% is perfect, degrades linearly
    if best_diff <= 0.03:
        return 1.0
    elif best_diff <= 0.06:
        return 0.8
    elif best_diff <= 0.10:
        return 0.5
    else:
        return max(0.0, 1.0 - best_diff * 5)


# Camelot wheel adjacency
_CAMELOT_NUMBERS = {}
for code, (root, mode) in {
    "1A": ("Ab", "minor"), "2A": ("Eb", "minor"), "3A": ("Bb", "minor"),
    "4A": ("F", "minor"), "5A": ("C", "minor"), "6A": ("G", "minor"),
    "7A": ("D", "minor"), "8A": ("A", "minor"), "9A": ("E", "minor"),
    "10A": ("B", "minor"), "11A": ("F#", "minor"), "12A": ("Db", "minor"),
    "1B": ("B", "major"), "2B": ("F#", "major"), "3B": ("Db", "major"),
    "4B": ("Ab", "major"), "5B": ("Eb", "major"), "6B": ("Bb", "major"),
    "7B": ("F", "major"), "8B": ("C", "major"), "9B": ("G", "major"),
    "10B": ("D", "major"), "11B": ("A", "major"), "12B": ("E", "major"),
}.items():
    _CAMELOT_NUMBERS[code] = (int(code[:-1]), code[-1])


def key_compatibility(camelot_a: str, camelot_b: str) -> float:
    """Score 0-1 based on Camelot wheel distance."""
    if camelot_a not in _CAMELOT_NUMBERS or camelot_b not in _CAMELOT_NUMBERS:
        return 0.0
    num_a, letter_a = _CAMELOT_NUMBERS[camelot_a]
    num_b, letter_b = _CAMELOT_NUMBERS[camelot_b]

    # Same position = perfect
    if camelot_a == camelot_b:
        return 1.0
    # Same number, different letter (relative major/minor) = great
    if num_a == num_b:
        return 0.9
    # Adjacent on wheel (+/- 1), same letter = great
    dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    if dist == 1 and letter_a == letter_b:
        return 0.85
    if dist == 1:
        return 0.6
    if dist == 2 and letter_a == letter_b:
        return 0.4
    return max(0.0, 1.0 - dist * 0.15)


def energy_compatibility(energy_a: float, energy_b: float) -> float:
    """Score 0-1 based on energy similarity (prefer smooth transitions)."""
    diff = abs(energy_a - energy_b)
    if diff <= 5:
        return 1.0
    elif diff <= 15:
        return 0.8
    elif diff <= 30:
        return 0.5
    else:
        return max(0.0, 1.0 - diff / 100)


def overall_score(bpm_s: float, key_s: float, energy_s: float) -> float:
    """Weighted overall compatibility score."""
    return bpm_s * 0.35 + key_s * 0.40 + energy_s * 0.25


# ── Track scanning & analysis ──────────────────────────────────────────


def scan_directory(directory: str) -> list[str]:
    """Find all supported audio files in directory."""
    files = []
    for entry in sorted(Path(directory).rglob("*")):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(entry))
    return files


def compute_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def try_load_analyzer():
    """Try to load essentia analyzer; fall back to librosa; else return None."""
    # Try essentia first
    try:
        from src.infrastructure.analyzers.essentia_analyzer import EssentiaAnalyzer
        analyzer = EssentiaAnalyzer()
        print("  [OK] Essentia analyzer loaded")
        return analyzer
    except ImportError:
        pass

    # Fallback: librosa-based analyzer
    try:
        import librosa  # noqa: F401
        print("  [OK] Librosa fallback loaded")
        return LibrosaFallbackAnalyzer()
    except ImportError:
        pass

    print("  [!!] No audio analyzer available")
    print("       Install essentia: pip install essentia")
    print("       Or librosa:      pip install librosa")
    return None


class LibrosaFallbackAnalyzer:
    """Minimal analyzer using librosa when essentia is not available."""

    def analyze(self, track):
        import uuid
        import librosa
        import numpy as np
        from src.domain.entities.analysis_result import AnalysisResult
        from src.domain.value_objects.bpm_value import BPMValue
        from src.domain.value_objects.energy_profile import EnergyProfile
        from src.domain.value_objects.key_signature import KeySignature

        y, sr = librosa.load(track.file_path, sr=22050, mono=True)

        # BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        bpm = BPMValue(round(float(tempo), 1))

        # Key detection via chroma + Krumhansl-Schmuckler profiles
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        # Major and minor key profiles (Krumhansl)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        pitch_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
        best_corr = -1
        best_root = "C"
        best_mode = "major"
        for shift in range(12):
            shifted = np.roll(chroma_avg, -shift)
            corr_maj = float(np.corrcoef(shifted, major_profile)[0, 1])
            corr_min = float(np.corrcoef(shifted, minor_profile)[0, 1])
            if corr_maj > best_corr:
                best_corr = corr_maj
                best_root = pitch_names[shift]
                best_mode = "major"
            if corr_min > best_corr:
                best_corr = corr_min
                best_root = pitch_names[shift]
                best_mode = "minor"
        key_notation = f"{best_root}m" if best_mode == "minor" else best_root
        key = KeySignature(key_notation)

        # Energy - use dB-scaled RMS, mapped to 0-100
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        # Map from [-80, 0] dB to [0, 100]
        overall = float(np.clip((np.mean(rms_db) + 80) / 80 * 100, 0, 100))

        # Segments (30s windows)
        segment_samples = 30 * sr
        segments = []
        for i in range(0, len(y), segment_samples):
            seg = y[i:i + segment_samples]
            seg_rms = librosa.feature.rms(y=seg)[0]
            seg_db = librosa.amplitude_to_db(seg_rms, ref=np.max)
            seg_energy = float(np.clip((np.mean(seg_db) + 80) / 80 * 100, 0, 100))
            segments.append({
                "start_ms": int(i / sr * 1000),
                "end_ms": int(min(i + segment_samples, len(y)) / sr * 1000),
                "level": round(seg_energy, 1),
            })

        if len(segments) >= 3:
            first = np.mean([s["level"] for s in segments[:len(segments) // 3]])
            last = np.mean([s["level"] for s in segments[-len(segments) // 3:]])
            trajectory = "build" if last > first * 1.1 else ("drop" if last < first * 0.9 else "maintain")
        else:
            trajectory = "maintain"

        energy = EnergyProfile(overall=round(overall, 1), segments=segments, trajectory=trajectory)

        return AnalysisResult(
            id=uuid.uuid4(),
            track_id=track.id,
            bpm=bpm,
            key=key,
            energy=energy,
        )


class DummyFingerprinter:
    """Fingerprinter that returns file hash as fingerprint."""
    def generate_fingerprint(self, file_path: str) -> str:
        return compute_hash(file_path)[:16]


class InMemoryTrackRepo:
    """Simple in-memory track repository for CLI testing."""

    def __init__(self):
        self._tracks = {}
        self._analysis = {}

    def save(self, track):
        self._tracks[str(track.id)] = track

    def find_by_id(self, track_id):
        return self._tracks.get(str(track_id))

    def find_by_hash(self, file_hash):
        for t in self._tracks.values():
            if t.file_hash == file_hash:
                return t
        return None

    def find_all(self):
        return list(self._tracks.values())

    def delete(self, track_id):
        self._tracks.pop(str(track_id), None)

    def save_analysis(self, result):
        self._analysis[str(result.track_id)] = result

    def find_analysis_by_track_id(self, track_id):
        return self._analysis.get(str(track_id))

    def all_analysis(self) -> list[AnalysisResult]:
        return list(self._analysis.values())


# ── Display helpers ────────────────────────────────────────────────────


def format_duration(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def track_display_name(file_path: str) -> str:
    return Path(file_path).stem


def print_track_table(tracks, analyses):
    """Print a formatted table of tracks."""
    print()
    print(f"  {'#':>3}  {'Title':<40} {'BPM':>6} {'Key':>4} {'Camelot':>7} {'Energy':>6} {'Trajectory':<10}")
    print(f"  {'─' * 3}  {'─' * 40} {'─' * 6} {'─' * 4} {'─' * 7} {'─' * 6} {'─' * 10}")

    for i, (track, analysis) in enumerate(zip(tracks, analyses)):
        name = track_display_name(track.file_path)[:40]
        if analysis:
            bpm = f"{analysis.bpm.value:.1f}"
            key = f"{analysis.key.root}{'m' if analysis.key.mode == 'minor' else ''}"
            camelot = analysis.key.to_camelot()
            energy = f"{analysis.energy.overall:.1f}"
            traj = analysis.energy.trajectory or "?"
        else:
            bpm = key = camelot = energy = traj = "..."

        print(f"  {i + 1:>3}  {name:<40} {bpm:>6} {key:>4} {camelot:>7} {energy:>6} {traj:<10}")
    print()


def print_recommendations(current_track, current_analysis, tracks, analyses):
    """Score and display recommendations."""
    if current_analysis is None:
        print("  Selected track has no analysis data.")
        return

    cur_bpm = current_analysis.bpm.value
    cur_camelot = current_analysis.key.to_camelot()
    cur_energy = current_analysis.energy.overall

    scored = []
    for track, analysis in zip(tracks, analyses):
        if str(track.id) == str(current_track.id) or analysis is None:
            continue
        bpm_s = bpm_compatibility(cur_bpm, analysis.bpm.value)
        key_s = key_compatibility(cur_camelot, analysis.key.to_camelot())
        energy_s = energy_compatibility(cur_energy, analysis.energy.overall)
        total = overall_score(bpm_s, key_s, energy_s)
        scored.append((track, analysis, total, bpm_s, key_s, energy_s))

    scored.sort(key=lambda x: x[2], reverse=True)

    cur_name = track_display_name(current_track.file_path)
    print(f"\n  Now Playing: {cur_name}")
    print(f"  BPM: {cur_bpm:.1f}  Key: {cur_camelot}  Energy: {cur_energy:.1f}")
    print()
    print(f"  {'#':>3}  {'Score':>5}  {'Title':<35} {'BPM':>6} {'Key':>7} {'Energy':>6}  {'BPM%':>4} {'Key%':>4} {'NRG%':>4}")
    print(f"  {'─' * 3}  {'─' * 5}  {'─' * 35} {'─' * 6} {'─' * 7} {'─' * 6}  {'─' * 4} {'─' * 4} {'─' * 4}")

    for i, (track, analysis, total, bpm_s, key_s, energy_s) in enumerate(scored[:10]):
        name = track_display_name(track.file_path)[:35]
        camelot = analysis.key.to_camelot()
        print(
            f"  {i + 1:>3}  {total:>5.0%}  {name:<35} "
            f"{analysis.bpm.value:>6.1f} {camelot:>7} {analysis.energy.overall:>6.1f}  "
            f"{bpm_s:>4.0%} {key_s:>4.0%} {energy_s:>4.0%}"
        )
    print()


def print_detail(track, analysis):
    """Print detailed analysis of a track."""
    name = track_display_name(track.file_path)
    print(f"\n  ── {name} ──")
    print(f"  File: {track.file_path}")
    print(f"  Hash: {track.file_hash[:16]}...")

    if analysis is None:
        print("  Analysis: NOT AVAILABLE")
        return

    key_str = f"{analysis.key.root} {'minor' if analysis.key.mode == 'minor' else 'major'}"
    print(f"  BPM:      {analysis.bpm.value:.1f}")
    print(f"  Key:      {key_str} ({analysis.key.to_camelot()})")
    print(f"  Energy:   {analysis.energy.overall:.1f} / 100  ({analysis.energy.trajectory})")

    if analysis.energy.segments:
        print(f"  Segments: {len(analysis.energy.segments)}")
        # Simple energy bar chart
        max_level = max(s["level"] for s in analysis.energy.segments) or 1
        for s in analysis.energy.segments:
            bar_len = int(s["level"] / max_level * 30)
            bar = "█" * bar_len
            start = s["start_ms"] // 1000
            print(f"    {start:>4}s  {bar} {s['level']:.1f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI DJ Assist - Analysis & Recommendation CLI")
    parser.add_argument("directory", help="Path to music directory")
    parser.add_argument("-n", "--limit", type=int, default=0, help="Max number of tracks to analyze (0 = all)")
    parser.add_argument("--random", action="store_true", help="Randomly select tracks (use with -n)")
    args = parser.parse_args()

    music_dir = os.path.expanduser(args.directory)
    max_tracks = args.limit
    use_random = args.random
    if not os.path.isdir(music_dir):
        print(f"Error: '{music_dir}' is not a directory")
        sys.exit(1)

    print("=" * 60)
    print("  AI DJ Assist - Analysis & Recommendation CLI")
    print("=" * 60)
    print()

    # Initialize
    print("[1/3] Loading analyzer...")
    analyzer = try_load_analyzer()
    if analyzer is None:
        sys.exit(1)

    repo = InMemoryTrackRepo()
    print("[2/3] Analyzer ready")

    # Scan and analyze
    def do_scan():
        print(f"[3/3] Scanning: {music_dir}")
        files = scan_directory(music_dir)
        total_found = len(files)
        if max_tracks > 0:
            if use_random:
                import random
                files = random.sample(files, min(max_tracks, len(files)))
            else:
                files = files[:max_tracks]
        print(f"  Found {total_found} audio files" + (f" (analyzing first {len(files)})" if max_tracks > 0 else ""))
        if not files:
            print("  No supported audio files found.")
            print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            return

        print()
        for i, fp in enumerate(files):
            name = track_display_name(fp)
            print(f"  Analyzing [{i + 1}/{len(files)}] {name}...", end="", flush=True)
            try:
                file_hash = compute_hash(fp)
                from src.domain.entities.audio_track import AudioTrack
                import uuid

                existing = repo.find_by_hash(file_hash)
                if existing and repo.find_analysis_by_track_id(existing.id):
                    print(" (cached)")
                    continue

                track = AudioTrack(id=uuid.uuid4(), file_path=fp, file_hash=file_hash)
                repo.save(track)
                result = analyzer.analyze(track)
                repo.save_analysis(result)
                analyzed = track.mark_as_analyzed()
                repo.save(analyzed)
                print(f" OK  BPM={result.bpm.value:.1f} Key={result.key.to_camelot()} Energy={result.energy.overall:.1f}")
            except Exception as e:
                print(f" FAIL: {e}")

    do_scan()

    # Interactive loop
    tracks = repo.find_all()
    analyses = [repo.find_analysis_by_track_id(t.id) for t in tracks]
    selected_idx = None

    print()
    print("─" * 60)
    print("Commands: list, select <#>, recommend, detail <#>, rescan, quit")
    print("─" * 60)

    while True:
        try:
            prompt = f"[{track_display_name(tracks[selected_idx].file_path)}]" if selected_idx is not None else "[no track]"
            cmd = input(f"\n  {prompt} > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0]
        arg = parts[1] if len(parts) > 1 else None

        if action in ("quit", "exit", "q"):
            print("Bye!")
            break

        elif action in ("list", "ls", "l"):
            tracks = repo.find_all()
            analyses = [repo.find_analysis_by_track_id(t.id) for t in tracks]
            print_track_table(tracks, analyses)

        elif action in ("select", "sel", "s"):
            if arg is None:
                print("  Usage: select <number>")
                continue
            try:
                idx = int(arg) - 1
                if 0 <= idx < len(tracks):
                    selected_idx = idx
                    t = tracks[idx]
                    a = analyses[idx]
                    name = track_display_name(t.file_path)
                    if a:
                        print(f"  ▶ Now playing: {name}  (BPM={a.bpm.value:.1f}  Key={a.key.to_camelot()}  Energy={a.energy.overall:.1f})")
                    else:
                        print(f"  ▶ Now playing: {name}  (no analysis)")
                else:
                    print(f"  Invalid number. Use 1-{len(tracks)}")
            except ValueError:
                print("  Usage: select <number>")

        elif action in ("recommend", "rec", "r"):
            if selected_idx is None:
                print("  Select a track first: select <number>")
                continue
            print_recommendations(tracks[selected_idx], analyses[selected_idx], tracks, analyses)

        elif action in ("detail", "info", "d"):
            if arg is None:
                if selected_idx is not None:
                    idx = selected_idx
                else:
                    print("  Usage: detail <number>")
                    continue
            else:
                try:
                    idx = int(arg) - 1
                except ValueError:
                    print("  Usage: detail <number>")
                    continue
            if 0 <= idx < len(tracks):
                print_detail(tracks[idx], analyses[idx])
            else:
                print(f"  Invalid number. Use 1-{len(tracks)}")

        elif action in ("rescan", "scan"):
            do_scan()
            tracks = repo.find_all()
            analyses = [repo.find_analysis_by_track_id(t.id) for t in tracks]
            print_track_table(tracks, analyses)

        elif action in ("help", "h", "?"):
            print("  Commands:")
            print("    list              Show all analyzed tracks")
            print("    select <#>        Select track as 'now playing'")
            print("    recommend         Get next track recommendations")
            print("    detail <#>        Detailed analysis of a track")
            print("    rescan            Re-scan music directory")
            print("    quit              Exit")

        else:
            print(f"  Unknown command: {action}. Type 'help' for commands.")


if __name__ == "__main__":
    main()
