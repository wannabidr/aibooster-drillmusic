from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# --- Optional FAISS (fallback provided) ---
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

# --- Audio IO + DSP ---
try:
    import soundfile as sf
except Exception as e:
    raise RuntimeError("soundfile가 필요합니다. `pip install soundfile`로 설치하세요.") from e

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa가 필요합니다. `pip install librosa`로 설치하세요.") from e

# --- Deep embedding (CLAP) ---
try:
    import torch
    from transformers import ClapModel, ClapProcessor
except Exception as e:
    raise RuntimeError(
        "torch/transformers가 필요합니다. `pip install torch transformers`로 설치하세요."
    ) from e


# =========================
# Utility / safety helpers
# =========================

EPS = 1e-12


def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    """Convert audio to mono float32 safely."""
    if y.ndim == 1:
        mono = y
    elif y.ndim == 2:
        # shape: (n_samples, n_channels) or (n_channels, n_samples)
        if y.shape[0] < y.shape[1]:
            # heuristics: channels first
            mono = np.mean(y, axis=0)
        else:
            mono = np.mean(y, axis=1)
    else:
        mono = y.reshape(-1)
    mono = mono.astype(np.float32, copy=False)
    # Normalize if integer PCM loaded unexpectedly
    if np.max(np.abs(mono)) > 1.5:
        mono = mono / (np.max(np.abs(mono)) + EPS)
    return mono


def load_audio(path: str, target_sr: Optional[int] = None, max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Robust audio loader.
    - Reads with soundfile
    - Converts to mono float32
    - Optional resample to target_sr using librosa
    - Optional truncate to max_seconds
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"오디오 파일이 없습니다: {path}")

    y, sr = sf.read(str(p), always_2d=False)
    y = _to_mono_float32(np.asarray(y))

    if max_seconds is not None and max_seconds > 0:
        max_len = int(sr * max_seconds)
        if y.shape[0] > max_len:
            y = y[:max_len]

    if target_sr is not None and sr != target_sr:
        # librosa.resample expects float
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr

    # avoid all-zero audio
    if float(np.max(np.abs(y))) < 1e-6:
        raise ValueError(f"오디오가 거의 무음입니다(또는 로딩 실패): {path}")

    return y, sr


def segment_audio(y: np.ndarray, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    """Return audio segment [start, start+dur). Clamps safely."""
    start = max(0, int(round(start_s * sr)))
    end = min(y.shape[0], int(round((start_s + dur_s) * sr)))
    if end <= start:
        return y[: min(y.shape[0], int(sr * dur_s))].copy()
    return y[start:end].copy()


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


# =========================
# Musical feature extraction
# =========================

_KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)

_KEY_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """
    Estimate BPM using librosa beat tracking.
    Returns BPM in float. Fallback to onset tempo if needed.
    """
    try:
        tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(tempo) or tempo <= 0:
            raise ValueError("invalid tempo")
        return tempo
    except Exception:
        # fallback: global tempo estimate
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempos = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        if tempos is None or len(tempos) == 0:
            return 120.0
        tempo = float(np.median(tempos))
        if not np.isfinite(tempo) or tempo <= 0:
            return 120.0
        return tempo


def estimate_key(y: np.ndarray, sr: int) -> Tuple[int, str, float]:
    """
    Estimate musical key from audio:
    - Compute chroma (CQT-based)
    - Compare with Krumhansl key profiles over 12 transpositions for major/minor
    Returns:
      root (0-11), mode ('major'/'minor'), tonal_clarity (0..1-ish)
    """
    # chroma_cqt is relatively robust for music; fallback to chroma_stft
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
    if float(np.sum(chroma_mean)) < EPS:
        return 0, "major", 0.0

    chroma_mean = chroma_mean / (float(np.sum(chroma_mean)) + EPS)

    major_profile = _KRUMHANSL_MAJOR / (float(np.sum(_KRUMHANSL_MAJOR)) + EPS)
    minor_profile = _KRUMHANSL_MINOR / (float(np.sum(_KRUMHANSL_MINOR)) + EPS)

    # correlations for each transposition
    major_scores = []
    minor_scores = []
    for shift in range(12):
        prof = np.roll(major_profile, shift)
        major_scores.append(float(np.dot(chroma_mean, prof)))
        profm = np.roll(minor_profile, shift)
        minor_scores.append(float(np.dot(chroma_mean, profm)))

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))
    major_best_score = major_scores[best_major]
    minor_best_score = minor_scores[best_minor]

    if major_best_score >= minor_best_score:
        root, mode, best = best_major, "major", major_best_score
        other = float(np.partition(np.array(major_scores), -2)[-2]) if len(major_scores) >= 2 else 0.0
    else:
        root, mode, best = best_minor, "minor", minor_best_score
        other = float(np.partition(np.array(minor_scores), -2)[-2]) if len(minor_scores) >= 2 else 0.0

    # tonal clarity: best vs 2nd-best margin (scaled)
    margin = max(0.0, best - other)
    clarity = float(margin / (best + EPS))
    clarity = float(np.clip(clarity, 0.0, 1.0))
    return root, mode, clarity


def estimate_energy(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Estimate perceived energy/intensity features.
    Returns dict with:
      rms, onset, centroid, energy_raw
    """
    # RMS
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(np.mean(onset_env))

    # Spectral centroid (brightness proxy)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_mean = float(np.mean(cent)) / float(sr / 2.0 + EPS)  # normalize 0..1

    # Combine (simple weighted sum)
    energy_raw = 0.50 * safe_log1p(rms_mean) + 0.35 * safe_log1p(onset_mean) + 0.15 * cent_mean
    return {
        "rms": float(rms_mean),
        "onset": float(onset_mean),
        "centroid": float(cent_mean),
        "energy_raw": float(energy_raw),
    }


def safe_log1p(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.log1p(max(0.0, x)))


def low_end_profile(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Low-end band energy ratios:
      sub(20-60), bass(60-120), lowmid(120-250)
    Uses STFT magnitude squared summed per band and normalized by total energy.
    """
    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    total = float(np.sum(S)) + EPS

    def band_ratio(f_lo: float, f_hi: float) -> float:
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if idx.size == 0:
            return 0.0
        return float(np.sum(S[idx, :]) / total)

    return {
        "sub": band_ratio(20, 60),
        "bass": band_ratio(60, 120),
        "lowmid": band_ratio(120, 250),
    }


def mixability_score(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Heuristic for 'mix-friendly intro/outro':
    - percussive_ratio: HPSS로 percussive 에너지 비율
    - stability: RMS 변동이 작을수록(= 안정적) 높게
    """
    # HPSS can be heavy, but reliable enough for 20~40s windows
    try:
        harm, perc = librosa.effects.hpss(y)
        harm_e = float(np.mean(harm**2))
        perc_e = float(np.mean(perc**2))
        percussive_ratio = float(perc_e / (harm_e + perc_e + EPS))
    except Exception:
        # fallback: treat as unknown
        percussive_ratio = 0.5

    rms = librosa.feature.rms(y=y)[0]
    mu = float(np.mean(rms)) + EPS
    sigma = float(np.std(rms))
    # stability: 높은 값이 안정적
    stability = float(np.clip(1.0 - (sigma / mu), 0.0, 1.0))

    return {
        "percussive_ratio": float(percussive_ratio),
        "stability": float(stability),
        "mixability": float(0.65 * percussive_ratio + 0.35 * stability),
    }


def key_compatibility(a_root: int, a_mode: str, b_root: int, b_mode: str) -> float:
    """
    Harmonic mixing compatibility score in [0,1].
    - 1.0: same key (root+mode)
    - 0.9: relative major/minor
    - 0.8: perfect fifth (±7 semitones) within same mode
    - 0.0: otherwise
    """
    a_root %= 12
    b_root %= 12
    if a_mode not in ("major", "minor") or b_mode not in ("major", "minor"):
        return 0.0

    if a_root == b_root and a_mode == b_mode:
        return 1.0

    # relative: C major (0) <-> A minor (9)
    # minor_root = (major_root - 3) mod 12  <=>  major_root = (minor_root + 3) mod 12
    if a_mode == "major" and b_mode == "minor" and b_root == (a_root - 3) % 12:
        return 0.9
    if a_mode == "minor" and b_mode == "major" and b_root == (a_root + 3) % 12:
        return 0.9

    # perfect fifth
    if a_mode == b_mode and ((b_root - a_root) % 12 in (7, 5)):  # +7 or -7
        return 0.8

    return 0.0


def bpm_compatible(a_bpm: float, b_bpm: float, tol: float = 4.0) -> bool:
    """
    BPM compatibility with half/double-time allowance.
    """
    if not np.isfinite(a_bpm) or not np.isfinite(b_bpm):
        return True
    if abs(a_bpm - b_bpm) <= tol:
        return True
    # half/double time
    if abs(a_bpm - 2.0 * b_bpm) <= tol:
        return True
    if abs(2.0 * a_bpm - b_bpm) <= tol:
        return True
    return False


# =========================
# CLAP Embedding
# =========================

class ClapEmbedder:
    """
    CLAP audio embedder wrapper.

    - Uses ClapProcessor + ClapModel.get_audio_features
    - Resamples input audio to 48kHz by default
    - Returns L2-normalized embedding vector (np.float32)
    """
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def target_sr(self) -> int:
        # CLAP processor는 sr 정보를 입력으로 받지 않기 때문에 48k로 맞춰주는 게 안전.
        return 48000

    @torch.no_grad()
    def embed_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr, res_type="kaiser_best")
            sr = self.target_sr

        # processor expects raw float array
        inputs = self.processor(audio=y, sampling_rate=self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        emb = self.model.get_audio_features(**inputs)  # shape: (1, dim)
        emb = emb.detach().float().cpu().numpy().reshape(-1)
        emb = l2_normalize(emb)
        return emb


# =========================
# Data structures
# =========================

@dataclass
class TrackInfo:
    track_id: str
    path: str
    duration_s: float

    bpm: float
    key_root: int
    key_mode: str
    tonal_clarity: float

    energy_raw: float
    sub: float
    bass: float
    lowmid: float

    intro_mix: float
    outro_mix: float


@dataclass
class IndexMeta:
    model_name: str
    intro_seconds: float
    outro_seconds: float
    bpm_tolerance: float

    # normalization stats for energy
    energy_mean: float
    energy_std: float

    tracks: List[TrackInfo]


# =========================
# Index build / save / load
# =========================

def compute_track_info(
    path: str,
    embedder: ClapEmbedder,
    intro_seconds: float,
    outro_seconds: float,
    analysis_sr: int = 22050,
) -> Tuple[TrackInfo, np.ndarray, np.ndarray]:
    """
    Compute:
    - TrackInfo (features)
    - intro embedding (for candidates)
    - outro embedding (for current->next matching)

    분석 특징은 22.05k로(속도/안정성), 임베딩은 48k로(모델 입력) 처리.
    """
    y_full, sr_full = load_audio(path, target_sr=None, max_seconds=None)
    duration_s = float(len(y_full) / sr_full)

    # analysis signal
    y_an, sr_an = (y_full, sr_full)
    if sr_full != analysis_sr:
        y_an = librosa.resample(y_full, orig_sr=sr_full, target_sr=analysis_sr, res_type="kaiser_best")
        sr_an = analysis_sr

    # Global features from (at most) first 120s for stability/speed
    y_global = y_an
    if duration_s > 120.0:
        y_global = segment_audio(y_an, sr_an, 0.0, 120.0)

    bpm = estimate_bpm(y_global, sr_an)
    key_root, key_mode, tonal_clarity = estimate_key(y_global, sr_an)

    e = estimate_energy(y_global, sr_an)
    low = low_end_profile(y_global, sr_an)

    # Intro/outro windows (analysis)
    intro_seg_an = segment_audio(y_an, sr_an, 0.0, min(intro_seconds, duration_s))
    outro_start = max(0.0, duration_s - outro_seconds)
    outro_seg_an = segment_audio(y_an, sr_an, outro_start, min(outro_seconds, duration_s))

    intro_mix = mixability_score(intro_seg_an, sr_an)["mixability"]
    outro_mix = mixability_score(outro_seg_an, sr_an)["mixability"]

    # Embeddings (use original sr to resample inside embedder)
    intro_seg = segment_audio(y_full, sr_full, 0.0, min(intro_seconds, duration_s))
    outro_seg = segment_audio(y_full, sr_full, max(0.0, duration_s - outro_seconds), min(outro_seconds, duration_s))
    intro_emb = embedder.embed_audio(intro_seg, sr_full)
    outro_emb = embedder.embed_audio(outro_seg, sr_full)

    track_id = Path(path).stem
    info = TrackInfo(
        track_id=track_id,
        path=str(Path(path).resolve()),
        duration_s=duration_s,
        bpm=float(bpm),
        key_root=int(key_root),
        key_mode=str(key_mode),
        tonal_clarity=float(tonal_clarity),
        energy_raw=float(e["energy_raw"]),
        sub=float(low["sub"]),
        bass=float(low["bass"]),
        lowmid=float(low["lowmid"]),
        intro_mix=float(intro_mix),
        outro_mix=float(outro_mix),
    )
    return info, intro_emb, outro_emb


def save_index(index_dir: str, meta: IndexMeta, intro_embs: np.ndarray, outro_embs: np.ndarray) -> None:
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(str(d / "intro_embeddings.npy"), intro_embs.astype(np.float32))
    np.save(str(d / "outro_embeddings.npy"), outro_embs.astype(np.float32))

    # Save meta
    meta_dict = dataclasses.asdict(meta)
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    # Build FAISS index if available
    if _FAISS_AVAILABLE:
        dim = int(intro_embs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(intro_embs.astype(np.float32))
        faiss.write_index(index, str(d / "faiss_intro.index"))


def load_index(index_dir: str) -> Tuple[IndexMeta, np.ndarray, Optional[object]]:
    d = Path(index_dir)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"index_dir에 meta.json이 없습니다: {index_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    tracks = [TrackInfo(**t) for t in meta_raw["tracks"]]
    meta = IndexMeta(
        model_name=meta_raw["model_name"],
        intro_seconds=float(meta_raw["intro_seconds"]),
        outro_seconds=float(meta_raw["outro_seconds"]),
        bpm_tolerance=float(meta_raw["bpm_tolerance"]),
        energy_mean=float(meta_raw["energy_mean"]),
        energy_std=float(meta_raw["energy_std"]),
        tracks=tracks,
    )

    intro_embs = np.load(str(d / "intro_embeddings.npy")).astype(np.float32)

    faiss_index = None
    if _FAISS_AVAILABLE:
        idx_path = d / "faiss_intro.index"
        if idx_path.exists():
            faiss_index = faiss.read_index(str(idx_path))

    return meta, intro_embs, faiss_index


def build_index(
    audio_dir: str,
    index_dir: str,
    model_name: str = "laion/clap-htsat-unfused",
    intro_seconds: float = 30.0,
    outro_seconds: float = 30.0,
    bpm_tolerance: float = 4.0,
) -> None:
    audio_dir_p = Path(audio_dir)
    if not audio_dir_p.exists():
        raise FileNotFoundError(f"audio_dir이 없습니다: {audio_dir}")

    # Collect audio files
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
    files = [p for p in audio_dir_p.rglob("*") if p.suffix.lower() in exts]
    if len(files) == 0:
        raise RuntimeError(f"오디오 파일을 찾지 못했습니다: {audio_dir}")

    embedder = ClapEmbedder(model_name=model_name)

    track_infos: List[TrackInfo] = []
    intro_embs: List[np.ndarray] = []
    outro_embs: List[np.ndarray] = []

    # Build
    errors: List[Tuple[str, str]] = []
    for p in tqdm(files, desc="Indexing tracks"):
        try:
            info, intro_e, outro_e = compute_track_info(
                path=str(p),
                embedder=embedder,
                intro_seconds=float(intro_seconds),
                outro_seconds=float(outro_seconds),
            )
            track_infos.append(info)
            intro_embs.append(intro_e)
            outro_embs.append(outro_e)
        except Exception as e:
            errors.append((str(p), f"{type(e).__name__}: {e}"))
            continue

    if len(track_infos) < 2:
        msg = "인덱스를 만들기에 성공한 트랙이 너무 적습니다(최소 2개 필요).\n"
        if errors:
            msg += "실패 목록(상위 5개):\n" + "\n".join([f"- {p}: {m}" for p, m in errors[:5]])
        raise RuntimeError(msg)

    intro_arr = np.stack(intro_embs, axis=0).astype(np.float32)
    outro_arr = np.stack(outro_embs, axis=0).astype(np.float32)

    # Energy normalization stats
    energies = np.array([t.energy_raw for t in track_infos], dtype=np.float32)
    e_mean = float(np.mean(energies))
    e_std = float(np.std(energies)) + 1e-6

    meta = IndexMeta(
        model_name=model_name,
        intro_seconds=float(intro_seconds),
        outro_seconds=float(outro_seconds),
        bpm_tolerance=float(bpm_tolerance),
        energy_mean=e_mean,
        energy_std=e_std,
        tracks=track_infos,
    )

    save_index(index_dir=index_dir, meta=meta, intro_embs=intro_arr, outro_embs=outro_arr)

    # Print summary (non-fatal errors)
    if errors:
        print(f"[WARN] {len(errors)}개 파일은 처리 실패(인덱스에서 제외). 예: {errors[0][0]} -> {errors[0][1]}", file=sys.stderr)


# =========================
# Recommendation
# =========================

def _search_candidates(
    query_emb: np.ndarray,
    intro_embs: np.ndarray,
    faiss_index: Optional[object],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, similarities) for top-k candidates.
    Similarity is cosine similarity (embeddings are normalized; inner product = cosine).
    """
    query = query_emb.reshape(1, -1).astype(np.float32)
    if faiss_index is not None:
        D, I = faiss_index.search(query, k)
        return I.reshape(-1), D.reshape(-1)

    # Fallback: brute-force cosine (still okay for small/medium libraries)
    sims = intro_embs @ query_emb.astype(np.float32)
    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return idx.astype(int), sims[idx].astype(np.float32)


def recommend_next(
    index_dir: str,
    current_path: str,
    goal: str = "maintain",  # maintain | up | down | peak
    top_k: int = 10,
    candidate_k: int = 200,
    history_ids: Optional[List[str]] = None,
    require_key: bool = True,
    min_tonal_clarity: float = 0.08,
) -> List[Dict[str, object]]:
    """
    Recommend next tracks given current track (raw audio path).
    Returns list of dicts with track info + score breakdown.

    goal:
      - maintain: energy similar
      - up: slightly higher energy
      - down: slightly lower energy
      - peak: much higher energy (aggressive)
    """
    meta, intro_embs, faiss_index = load_index(index_dir)

    embedder = ClapEmbedder(model_name=meta.model_name)

    # Analyze current track
    y_full, sr_full = load_audio(current_path, target_sr=None, max_seconds=None)
    duration_s = float(len(y_full) / sr_full)

    outro_start = max(0.0, duration_s - meta.outro_seconds)
    outro_seg = segment_audio(y_full, sr_full, outro_start, meta.outro_seconds)

    # Query embedding
    q_emb = embedder.embed_audio(outro_seg, sr_full)

    # Current features (for constraints)
    # analysis at 22.05k
    y_an = y_full
    if sr_full != 22050:
        y_an = librosa.resample(y_full, orig_sr=sr_full, target_sr=22050, res_type="kaiser_best")
        sr_an = 22050
    else:
        sr_an = sr_full

    y_global = y_an if duration_s <= 120.0 else segment_audio(y_an, sr_an, 0.0, 120.0)
    cur_bpm = estimate_bpm(y_global, sr_an)
    cur_root, cur_mode, cur_clarity = estimate_key(y_global, sr_an)
    cur_energy = estimate_energy(y_global, sr_an)["energy_raw"]
    cur_low = low_end_profile(y_global, sr_an)

    # Candidate search
    candidate_k = int(max(top_k, candidate_k))
    idxs, sims = _search_candidates(q_emb, intro_embs, faiss_index, k=min(candidate_k, len(meta.tracks)))

    history = set(history_ids or [])
    results: List[Tuple[float, Dict[str, object]]] = []

    # Weighting (tunable)
    w_embed = 1.00
    w_energy = 0.35
    w_low = 0.25
    w_mix = 0.20
    w_key = 0.25  # used only when key constraints enabled

    # Normalize current energy
    cur_ez = (cur_energy - meta.energy_mean) / (meta.energy_std + EPS)

    for rank_pos, (i, sim) in enumerate(zip(idxs, sims)):
        if i < 0 or i >= len(meta.tracks):
            continue
        t = meta.tracks[int(i)]

        # Avoid same track / history
        if t.track_id in history:
            continue
        if Path(t.path).resolve() == Path(current_path).resolve():
            continue

        # Hard constraints: BPM
        if not bpm_compatible(cur_bpm, t.bpm, tol=meta.bpm_tolerance):
            continue

        # Hard-ish constraints: key (only if tonal clarity is good)
        key_ok = True
        key_score = 0.0
        if require_key and (cur_clarity >= min_tonal_clarity) and (t.tonal_clarity >= min_tonal_clarity):
            key_score = key_compatibility(cur_root, cur_mode, t.key_root, t.key_mode)
            key_ok = key_score > 0.0
            if not key_ok:
                continue

        # Soft score components
        embed_score = float(sim)  # cosine similarity (higher better)

        # Energy target scoring
        t_ez = (t.energy_raw - meta.energy_mean) / (meta.energy_std + EPS)
        if goal == "maintain":
            energy_score = float(np.exp(-abs(t_ez - cur_ez)))
        elif goal == "up":
            # prefer slightly higher
            delta = float(t_ez - cur_ez)
            energy_score = float(np.exp(-abs(delta - 0.5)))  # target +0.5 z
        elif goal == "down":
            delta = float(t_ez - cur_ez)
            energy_score = float(np.exp(-abs(delta + 0.5)))  # target -0.5 z
        elif goal == "peak":
            delta = float(t_ez - cur_ez)
            energy_score = float(np.exp(-abs(delta - 1.2)))  # target +1.2 z
        else:
            energy_score = float(np.exp(-abs(t_ez - cur_ez)))

        # Low-end compatibility (reduce collision risk)
        # penalize big differences in sub/bass ratios
        low_dist = abs(t.sub - cur_low["sub"]) + 0.7 * abs(t.bass - cur_low["bass"]) + 0.4 * abs(t.lowmid - cur_low["lowmid"])
        low_score = float(np.exp(-3.0 * low_dist))

        # Mixability: we want candidate intro to be mix-friendly; also prefer current outro being mix-friendly (but fixed)
        mix_score = float(np.clip(t.intro_mix, 0.0, 1.0))

        # Key score (already filtered). If key not used, keep 0.
        if require_key:
            if (cur_clarity >= min_tonal_clarity) and (t.tonal_clarity >= min_tonal_clarity):
                kscore = float(key_score)
            else:
                # if key unclear, don't punish
                kscore = 0.5
        else:
            kscore = 0.0

        total = (
            w_embed * embed_score
            + w_energy * energy_score
            + w_low * low_score
            + w_mix * mix_score
            + (w_key * kscore if require_key else 0.0)
        )

        detail = {
            "track_id": t.track_id,
            "path": t.path,
            "score": float(total),
            "components": {
                "embed": float(embed_score),
                "energy": float(energy_score),
                "low_end": float(low_score),
                "mix_intro": float(mix_score),
                "key": float(kscore) if require_key else None,
            },
            "features": {
                "bpm": float(t.bpm),
                "key": f"{_KEY_NAMES_SHARP[t.key_root]} {t.key_mode}",
                "tonal_clarity": float(t.tonal_clarity),
                "energy_z": float(t_ez),
                "sub": float(t.sub),
                "bass": float(t.bass),
                "intro_mix": float(t.intro_mix),
            },
            "rank_in_candidates": int(rank_pos),
        }
        results.append((float(total), detail))

    # Sort and return top_k
    results.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in results[: int(top_k)]]


# =========================
# Continuous session helper
# =========================

class SessionRecommender:
    """
    Keeps history and provides continuous recommendations.
    """
    def __init__(
        self,
        index_dir: str,
        history_limit: int = 50,
        require_key: bool = True,
        min_tonal_clarity: float = 0.08,
    ):
        self.index_dir = index_dir
        self.history_limit = int(history_limit)
        self.require_key = require_key
        self.min_tonal_clarity = float(min_tonal_clarity)
        self._history: List[str] = []

    def add_played(self, track_id: str) -> None:
        self._history.append(track_id)
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit :]

    def next(
        self,
        current_path: str,
        goal: str = "maintain",
        top_k: int = 10,
        candidate_k: int = 200,
    ) -> List[Dict[str, object]]:
        return recommend_next(
            index_dir=self.index_dir,
            current_path=current_path,
            goal=goal,
            top_k=top_k,
            candidate_k=candidate_k,
            history_ids=self._history,
            require_key=self.require_key,
            min_tonal_clarity=self.min_tonal_clarity,
        )


# =========================
# CLI
# =========================

def _cmd_build(args: argparse.Namespace) -> None:
    build_index(
        audio_dir=args.audio_dir,
        index_dir=args.index_dir,
        model_name=args.model,
        intro_seconds=args.intro_seconds,
        outro_seconds=args.outro_seconds,
        bpm_tolerance=args.bpm_tolerance,
    )
    print(f"[OK] Index built at: {args.index_dir}")


def _cmd_recommend(args: argparse.Namespace) -> None:
    out = recommend_next(
        index_dir=args.index_dir,
        current_path=args.current,
        goal=args.goal,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        history_ids=args.history_ids,
        require_key=not args.no_key,
        min_tonal_clarity=args.min_tonal_clarity,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Raw-audio DJ-style next-track recommender (CLAP + constraints)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build index from an audio folder")
    b.add_argument("--audio_dir", required=True, help="Folder containing audio files")
    b.add_argument("--index_dir", required=True, help="Output index directory")
    b.add_argument("--model", default="laion/clap-htsat-unfused", help="HF model name for CLAP")
    b.add_argument("--intro_seconds", type=float, default=30.0, help="Intro window seconds for candidate embedding")
    b.add_argument("--outro_seconds", type=float, default=30.0, help="Outro window seconds for query embedding")
    b.add_argument("--bpm_tolerance", type=float, default=4.0, help="BPM tolerance (also checks half/double-time)")
    b.set_defaults(func=_cmd_build)

    r = sub.add_parser("recommend", help="Recommend next tracks for a current track")
    r.add_argument("--index_dir", required=True, help="Index directory")
    r.add_argument("--current", required=True, help="Current audio file path")
    r.add_argument("--goal", default="maintain", choices=["maintain", "up", "down", "peak"], help="Energy curve goal")
    r.add_argument("--top_k", type=int, default=10, help="Return top-K recommendations")
    r.add_argument("--candidate_k", type=int, default=200, help="Initial ANN candidate size before filtering/rerank")
    r.add_argument("--history_ids", nargs="*", default=None, help="Track IDs to exclude (already played)")
    r.add_argument("--no_key", action="store_true", help="Disable key constraint / scoring")
    r.add_argument("--min_tonal_clarity", type=float, default=0.08, help="Minimum key clarity to enforce key rules")
    r.set_defaults(func=_cmd_recommend)

    return p


def main() -> None:
    try:
        parser = make_argparser()
        args = parser.parse_args()
        args.func(args)
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Friendly error output
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        # If user wants deeper debug, show trace with env var
        if os.environ.get("DJREC_DEBUG", "").strip() == "1":
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()