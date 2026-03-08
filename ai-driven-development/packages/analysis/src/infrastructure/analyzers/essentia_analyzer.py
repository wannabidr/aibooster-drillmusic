"""Essentia-based audio analyzer implementation."""

from __future__ import annotations

import uuid

import essentia.standard as es
import numpy as np

from src.domain.entities.analysis_result import AnalysisResult
from src.domain.entities.audio_track import AudioTrack
from src.domain.ports.audio_analyzer import AudioAnalyzer
from src.domain.value_objects.bpm_value import BPMValue
from src.domain.value_objects.energy_profile import EnergyProfile
from src.domain.value_objects.key_signature import KeySignature

# Essentia key output -> standard notation mapping
_ESSENTIA_KEY_MAP: dict[str, str] = {
    "A minor": "Am",
    "A major": "A",
    "Bb minor": "Bbm",
    "Bb major": "Bb",
    "B minor": "Bm",
    "B major": "B",
    "C minor": "Cm",
    "C major": "C",
    "C# minor": "Dbm",
    "Db minor": "Dbm",
    "C# major": "Db",
    "Db major": "Db",
    "D minor": "Dm",
    "D major": "D",
    "Eb minor": "Ebm",
    "Eb major": "Eb",
    "E minor": "Em",
    "E major": "E",
    "F minor": "Fm",
    "F major": "F",
    "F# minor": "F#m",
    "F# major": "F#",
    "G minor": "Gm",
    "G major": "G",
    "Ab minor": "Abm",
    "Ab major": "Ab",
}


class EssentiaAnalyzer(AudioAnalyzer):
    def __init__(self, sample_rate: int = 44100) -> None:
        self._sample_rate = sample_rate

    def analyze(self, track: AudioTrack) -> AnalysisResult:
        audio = self._load_audio(track.file_path)
        bpm = self._detect_bpm(audio)
        key = self._detect_key(audio)
        energy = self._compute_energy(audio)

        return AnalysisResult(
            id=uuid.uuid4(),
            track_id=track.id,
            bpm=bpm,
            key=key,
            energy=energy,
        )

    def _load_audio(self, file_path: str) -> np.ndarray:
        loader = es.MonoLoader(filename=file_path, sampleRate=self._sample_rate)
        return loader()

    def _detect_bpm(self, audio: np.ndarray) -> BPMValue:
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _beats, _confidence, _estimates, _intervals = rhythm_extractor(audio)
        return BPMValue(round(float(bpm), 1))

    def _detect_key(self, audio: np.ndarray) -> KeySignature:
        key_extractor = es.KeyExtractor()
        key, scale, _strength = key_extractor(audio)
        essentia_key = f"{key} {scale}"
        notation = _ESSENTIA_KEY_MAP.get(essentia_key, "C")
        return KeySignature(notation)

    def _compute_energy(self, audio: np.ndarray) -> EnergyProfile:
        # Overall RMS energy normalized to 0-100
        rms = float(np.sqrt(np.mean(audio**2)))
        overall = min(100.0, rms * 1000)

        # Segment energy (30-second windows)
        segment_duration = 30 * self._sample_rate
        segments = []
        for i in range(0, len(audio), segment_duration):
            segment = audio[i : i + segment_duration]
            seg_rms = float(np.sqrt(np.mean(segment**2)))
            seg_energy = min(100.0, seg_rms * 1000)
            segments.append(
                {
                    "start_ms": int(i / self._sample_rate * 1000),
                    "end_ms": int(min(i + segment_duration, len(audio)) / self._sample_rate * 1000),
                    "level": round(seg_energy, 1),
                }
            )

        # Determine trajectory from first and last third
        if len(segments) >= 3:
            first_third = np.mean([s["level"] for s in segments[: len(segments) // 3]])
            last_third = np.mean([s["level"] for s in segments[-len(segments) // 3 :]])
            if last_third > first_third * 1.1:
                trajectory = "build"
            elif last_third < first_third * 0.9:
                trajectory = "drop"
            else:
                trajectory = "maintain"
        else:
            trajectory = "maintain"

        return EnergyProfile(
            overall=round(overall, 1),
            segments=segments,
            trajectory=trajectory,
        )
