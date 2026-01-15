"""
main.py (DJ Recommender)
=======================
Raw audio only로 "다음 곡 자동 추천(연속 추천)"을 구현하는 단일 파일 베이스라인.

✅ 반영된 요소
- 딥러닝 임베딩: Hugging Face CLAP로 intro/outro 임베딩 추출
- 조건 기반 필터: BPM, Key(tonal clarity 기반), 저역 충돌
- 흥(그루브/텐션) 반영: onset_density(추진력), percussive_ratio(드럼성), energy
- 다양성 반영: 최근 히스토리와 임베딩/보컬/밝기/저역 성향의 반복 방지 및 완만한 변주 유도
- Windows/MP3/리샘플 안정성 강화:
  - soundfile 실패 시 librosa.load fallback
  - resampy 없이도 동작하도록 scipy 기반 resample_poly 사용

사용 예
- 인덱스 생성:
    python main.py build --audio_dir ../../Music/basshouse --index_dir ./index
- 추천:
    python main.py recommend --index_dir ./index --current path/to/current.mp3 --goal up --top_k 10

주의
- TrackInfo/IndexMeta 확장되어 기존 index_dir과 호환 안 될 수 있습니다. 새로 build를 권장합니다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Optional FAISS (fallback provided)
try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

# Audio IO + DSP
try:
    import soundfile as sf
except Exception as e:
    raise RuntimeError("soundfile가 필요합니다. `pip install soundfile`로 설치하세요.") from e

try:
    import librosa
except Exception as e:
    raise RuntimeError("librosa가 필요합니다. `pip install librosa`로 설치하세요.") from e

try:
    from scipy.signal import resample_poly
except Exception as e:
    raise RuntimeError("scipy가 필요합니다. `pip install scipy`로 설치하세요.") from e

# Deep embedding (CLAP)
try:
    import torch
    from transformers import ClapModel, ClapProcessor
except Exception as e:
    raise RuntimeError("torch/transformers가 필요합니다. `pip install torch transformers`로 설치하세요.") from e


EPS = 1e-12


# -------------------------
# Robust resampling (no resampy dependency)
# -------------------------
def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    scipy.signal.resample_poly 기반의 안정적인 리샘플러.
    - resampy/soxr 미설치 환경에서도 동작.
    """
    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)

    # Reduce fraction by gcd to keep small integers
    g = int(np.gcd(orig_sr, target_sr))
    up = target_sr // g
    down = orig_sr // g

    y = y.astype(np.float32, copy=False)
    y2 = resample_poly(y, up=up, down=down).astype(np.float32, copy=False)

    # Guard for all-zero due to weird input
    if float(np.max(np.abs(y2))) < 1e-7:
        # fallback: librosa resample if available (may still work)
        try:
            y2 = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32, copy=False)
        except Exception:
            pass
    return y2


def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    """Convert audio to mono float32 safely."""
    if y.ndim == 1:
        mono = y
    elif y.ndim == 2:
        # shape: (n_samples, n_channels) or (n_channels, n_samples)
        if y.shape[0] < y.shape[1]:
            mono = np.mean(y, axis=0)
        else:
            mono = np.mean(y, axis=1)
    else:
        mono = y.reshape(-1)

    mono = mono.astype(np.float32, copy=False)

    # Normalize if integer PCM or large scale
    m = float(np.max(np.abs(mono))) + EPS
    if m > 1.5:
        mono = mono / m
    return mono


def load_audio(path: str, target_sr: Optional[int] = None, max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Robust audio loader.
    1) soundfile로 시도
    2) 실패 시 librosa.load fallback (ffmpeg/audioread 환경이면 MP3 성공률↑)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"오디오 파일이 없습니다: {path}")

    y = None
    sr = None

    # 1) Try soundfile
    try:
        y_sf, sr_sf = sf.read(str(p), always_2d=False)
        y = _to_mono_float32(np.asarray(y_sf))
        sr = int(sr_sf)
    except Exception:
        # 2) Fallback to librosa.load
        y_lb, sr_lb = librosa.load(str(p), sr=None, mono=True)
        y = _to_mono_float32(np.asarray(y_lb))
        sr = int(sr_lb)

    if max_seconds is not None and max_seconds > 0:
        max_len = int(sr * max_seconds)
        if y.shape[0] > max_len:
            y = y[:max_len]

    if target_sr is not None and sr != target_sr:
        y = resample_audio(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if float(np.max(np.abs(y))) < 1e-6:
        raise ValueError(f"오디오가 거의 무음입니다(또는 로딩 실패): {path}")

    return y, sr


def segment_audio(y: np.ndarray, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    """Return audio segment [start, start+dur). Clamps safely."""
    start = max(0, int(round(start_s * sr)))
    end = min(y.shape[0], int(round((start_s + dur_s) * sr)))
    if end <= start:
        # fallback: first dur_s seconds
        end2 = min(y.shape[0], int(sr * dur_s))
        return y[:end2].copy()
    return y[start:end].copy()


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


# -------------------------
# Musical feature extraction
# -------------------------
_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)
_KEY_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def safe_log1p(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.log1p(max(0.0, x)))


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """
    Estimate BPM using librosa beat tracking.
    Numpy 1.25+ 경고 방지 위해 tempo가 array면 첫 원소 사용.
    """
    try:
        tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(tempo) or tempo <= 0:
            raise ValueError("invalid tempo")
        return tempo
    except Exception:
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
    Key 추정 + tonal clarity.
    - chroma(CQT or STFT) -> 평균 -> Krumhansl 프로파일 dot-product
    """
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

    major_scores = []
    minor_scores = []
    for shift in range(12):
        major_scores.append(float(np.dot(chroma_mean, np.roll(major_profile, shift))))
        minor_scores.append(float(np.dot(chroma_mean, np.roll(minor_profile, shift))))

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))
    major_best = float(major_scores[best_major])
    minor_best = float(minor_scores[best_minor])

    if major_best >= minor_best:
        root, mode, best = best_major, "major", major_best
        other = float(np.partition(np.array(major_scores, dtype=np.float32), -2)[-2]) if len(major_scores) >= 2 else 0.0
    else:
        root, mode, best = best_minor, "minor", minor_best
        other = float(np.partition(np.array(minor_scores, dtype=np.float32), -2)[-2]) if len(minor_scores) >= 2 else 0.0

    margin = max(0.0, best - other)
    clarity = float(np.clip(margin / (best + EPS), 0.0, 1.0))
    return int(root), str(mode), clarity


def estimate_energy(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    에너지(텐션) proxy:
    RMS(음량) + onset strength(타격) + spectral centroid(밝기)
    """
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_mean = float(np.mean(onset_env))

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_mean = float(np.mean(cent)) / float(sr / 2.0 + EPS)

    energy_raw = 0.50 * safe_log1p(rms_mean) + 0.35 * safe_log1p(onset_mean) + 0.15 * cent_mean
    return {"rms": rms_mean, "onset": onset_mean, "centroid": cent_mean, "energy_raw": energy_raw}


def low_end_profile(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    저역 밴드 비율:
      sub(20-60), bass(60-120), lowmid(120-250)
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

    return {"sub": band_ratio(20, 60), "bass": band_ratio(60, 120), "lowmid": band_ratio(120, 250)}


def mixability_score(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    믹스 용이성(인트로/아웃트로):
    - percussive_ratio (HPSS)
    - stability (RMS 변동 적을수록)
    """
    try:
        harm, perc = librosa.effects.hpss(y)
        harm_e = float(np.mean(harm**2))
        perc_e = float(np.mean(perc**2))
        percussive_ratio = float(perc_e / (harm_e + perc_e + EPS))
    except Exception:
        percussive_ratio = 0.5

    rms = librosa.feature.rms(y=y)[0]
    mu = float(np.mean(rms)) + EPS
    sigma = float(np.std(rms))
    stability = float(np.clip(1.0 - (sigma / mu), 0.0, 1.0))

    return {
        "percussive_ratio": percussive_ratio,
        "stability": stability,
        "mixability": float(0.65 * percussive_ratio + 0.35 * stability),
    }


def estimate_onset_density(y: np.ndarray, sr: int) -> float:
    """단위 시간당 온셋 개수(이벤트/추진력)."""
    dur = float(len(y) / sr)
    if dur <= 0.5:
        return 0.0
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        return float(len(onsets) / dur)
    except Exception:
        return 0.0


def estimate_brightness(y: np.ndarray, sr: int) -> float:
    """밝기(정규화된 spectral centroid 평균)."""
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        cent_mean = float(np.mean(cent))
        return float(np.clip(cent_mean / (sr / 2.0 + EPS), 0.0, 1.0))
    except Exception:
        return 0.0


def estimate_vocal_presence(y: np.ndarray, sr: int) -> float:
    """
    보컬/훅 존재감 proxy:
    - 300~3400Hz 에너지 비율
    """
    try:
        n_fft = 2048
        hop = 512
        S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        total_idx = np.where((freqs >= 80) & (freqs <= 8000))[0]
        voc_idx = np.where((freqs >= 300) & (freqs <= 3400))[0]
        total = float(np.sum(S[total_idx, :])) + EPS
        voc = float(np.sum(S[voc_idx, :]))
        return float(np.clip(voc / total, 0.0, 1.0))
    except Exception:
        return 0.0


def estimate_percussive_ratio_global(y: np.ndarray, sr: int) -> float:
    """전역 퍼커시브 비율(HPSS)."""
    try:
        harm, perc = librosa.effects.hpss(y)
        harm_e = float(np.mean(harm**2))
        perc_e = float(np.mean(perc**2))
        return float(np.clip(perc_e / (harm_e + perc_e + EPS), 0.0, 1.0))
    except Exception:
        return 0.5


def key_compatibility(a_root: int, a_mode: str, b_root: int, b_mode: str) -> float:
    """
    하모닉 믹싱 호환도 [0..1]
    - 1.0: same key+mode
    - 0.9: relative major/minor
    - 0.8: perfect fifth (±7) same mode
    """
    a_root %= 12
    b_root %= 12
    if a_mode not in ("major", "minor") or b_mode not in ("major", "minor"):
        return 0.0

    if a_root == b_root and a_mode == b_mode:
        return 1.0

    # relative major/minor
    if a_mode == "major" and b_mode == "minor" and b_root == (a_root - 3) % 12:
        return 0.9
    if a_mode == "minor" and b_mode == "major" and b_root == (a_root + 3) % 12:
        return 0.9

    # perfect fifth (±7 semitones)
    if a_mode == b_mode and ((b_root - a_root) % 12 in (7, 5)):
        return 0.8

    return 0.0


def bpm_compatible(a_bpm: float, b_bpm: float, tol: float = 4.0) -> bool:
    """BPM 호환 + half/double time 허용."""
    if not np.isfinite(a_bpm) or not np.isfinite(b_bpm):
        return True
    if abs(a_bpm - b_bpm) <= tol:
        return True
    if abs(a_bpm - 2.0 * b_bpm) <= tol:
        return True
    if abs(2.0 * a_bpm - b_bpm) <= tol:
        return True
    return False


# -------------------------
# CLAP Embedding
# -------------------------
class ClapEmbedder:
    """
    CLAP audio embedder wrapper.

    - ClapProcessor + ClapModel.get_audio_features
    - 입력을 48kHz로 통일
    - processor에는 audio=..., sampling_rate=... 전달 (경고 제거 + 안정성)
    """

    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @property
    def target_sr(self) -> int:
        return 48000

    @torch.no_grad()
    def embed_audio(self, y: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            y = resample_audio(y, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        inputs = self.processor(audio=y, sampling_rate=self.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        emb = self.model.get_audio_features(**inputs)  # (1, dim)
        emb = emb.detach().float().cpu().numpy().reshape(-1)
        return l2_normalize(emb)


# -------------------------
# Data structures
# -------------------------
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

    # diversity/hype features
    onset_density: float  # events/sec
    brightness: float  # 0..1
    vocal_presence: float  # 0..1
    perc_ratio_global: float  # 0..1


@dataclass
class IndexMeta:
    model_name: str
    intro_seconds: float
    outro_seconds: float
    bpm_tolerance: float

    energy_mean: float
    energy_std: float

    onset_mean: float
    onset_std: float
    bright_mean: float
    bright_std: float
    vocal_mean: float
    vocal_std: float
    perc_mean: float
    perc_std: float

    tracks: List[TrackInfo]


# -------------------------
# Index build / save / load
# -------------------------
def compute_track_info(
    path: str,
    embedder: ClapEmbedder,
    intro_seconds: float,
    outro_seconds: float,
    analysis_sr: int = 22050,
) -> Tuple[TrackInfo, np.ndarray, np.ndarray]:
    """
    한 곡 분석:
    - 전역 특징(최대 120s) + intro/outro mixability + intro/outro CLAP 임베딩
    """
    y_full, sr_full = load_audio(path, target_sr=None, max_seconds=None)
    duration_s = float(len(y_full) / sr_full)

    # analysis signal @ 22.05k
    y_an, sr_an = (y_full, sr_full)
    if sr_full != analysis_sr:
        y_an = resample_audio(y_full, orig_sr=sr_full, target_sr=analysis_sr)
        sr_an = analysis_sr

    # global window (first 120s)
    y_global = y_an
    if duration_s > 120.0:
        y_global = segment_audio(y_an, sr_an, 0.0, 120.0)

    bpm = estimate_bpm(y_global, sr_an)
    key_root, key_mode, tonal_clarity = estimate_key(y_global, sr_an)

    e = estimate_energy(y_global, sr_an)
    low = low_end_profile(y_global, sr_an)

    # NEW global hype/diversity features
    onset_density = estimate_onset_density(y_global, sr_an)
    brightness = estimate_brightness(y_global, sr_an)
    vocal_presence = estimate_vocal_presence(y_global, sr_an)
    perc_ratio_global = estimate_percussive_ratio_global(y_global, sr_an)

    # intro/outro segments for mixability (analysis sr)
    intro_seg_an = segment_audio(y_an, sr_an, 0.0, min(intro_seconds, duration_s))
    outro_start = max(0.0, duration_s - outro_seconds)
    outro_seg_an = segment_audio(y_an, sr_an, outro_start, min(outro_seconds, duration_s))

    intro_mix = mixability_score(intro_seg_an, sr_an)["mixability"]
    outro_mix = mixability_score(outro_seg_an, sr_an)["mixability"]

    # embeddings (use original sr; embedder will resample to 48k)
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
        onset_density=float(onset_density),
        brightness=float(brightness),
        vocal_presence=float(vocal_presence),
        perc_ratio_global=float(perc_ratio_global),
    )
    return info, intro_emb, outro_emb


def save_index(index_dir: str, meta: IndexMeta, intro_embs: np.ndarray, outro_embs: np.ndarray) -> None:
    d = Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)

    np.save(str(d / "intro_embeddings.npy"), intro_embs.astype(np.float32))
    np.save(str(d / "outro_embeddings.npy"), outro_embs.astype(np.float32))

    meta_dict = dataclasses.asdict(meta)
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    if _FAISS_AVAILABLE:
        dim = int(intro_embs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(intro_embs.astype(np.float32))
        faiss.write_index(index, str(d / "faiss_intro.index"))


def _get(meta_raw: dict, key: str, default: float) -> float:
    """Backward-compat helper (meta.json에 필드 없을 때 기본값)."""
    v = meta_raw.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


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
        intro_seconds=_get(meta_raw, "intro_seconds", 30.0),
        outro_seconds=_get(meta_raw, "outro_seconds", 30.0),
        bpm_tolerance=_get(meta_raw, "bpm_tolerance", 4.0),
        energy_mean=_get(meta_raw, "energy_mean", 0.0),
        energy_std=_get(meta_raw, "energy_std", 1.0),
        onset_mean=_get(meta_raw, "onset_mean", 0.0),
        onset_std=_get(meta_raw, "onset_std", 1.0),
        bright_mean=_get(meta_raw, "bright_mean", 0.0),
        bright_std=_get(meta_raw, "bright_std", 1.0),
        vocal_mean=_get(meta_raw, "vocal_mean", 0.0),
        vocal_std=_get(meta_raw, "vocal_std", 1.0),
        perc_mean=_get(meta_raw, "perc_mean", 0.0),
        perc_std=_get(meta_raw, "perc_std", 1.0),
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

    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
    files = [p for p in audio_dir_p.rglob("*") if p.suffix.lower() in exts]
    if len(files) == 0:
        raise RuntimeError(f"오디오 파일을 찾지 못했습니다: {audio_dir}")

    embedder = ClapEmbedder(model_name=model_name)

    track_infos: List[TrackInfo] = []
    intro_embs: List[np.ndarray] = []
    outro_embs: List[np.ndarray] = []

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

    energies = np.array([t.energy_raw for t in track_infos], dtype=np.float32)
    e_mean = float(np.mean(energies))
    e_std = float(np.std(energies)) + 1e-6

    onset_arr = np.array([t.onset_density for t in track_infos], dtype=np.float32)
    bright_arr = np.array([t.brightness for t in track_infos], dtype=np.float32)
    vocal_arr = np.array([t.vocal_presence for t in track_infos], dtype=np.float32)
    perc_arr = np.array([t.perc_ratio_global for t in track_infos], dtype=np.float32)

    onset_mean, onset_std = float(np.mean(onset_arr)), float(np.std(onset_arr)) + 1e-6
    bright_mean, bright_std = float(np.mean(bright_arr)), float(np.std(bright_arr)) + 1e-6
    vocal_mean, vocal_std = float(np.mean(vocal_arr)), float(np.std(vocal_arr)) + 1e-6
    perc_mean, perc_std = float(np.mean(perc_arr)), float(np.std(perc_arr)) + 1e-6

    meta = IndexMeta(
        model_name=model_name,
        intro_seconds=float(intro_seconds),
        outro_seconds=float(outro_seconds),
        bpm_tolerance=float(bpm_tolerance),
        energy_mean=e_mean,
        energy_std=e_std,
        onset_mean=onset_mean,
        onset_std=onset_std,
        bright_mean=bright_mean,
        bright_std=bright_std,
        vocal_mean=vocal_mean,
        vocal_std=vocal_std,
        perc_mean=perc_mean,
        perc_std=perc_std,
        tracks=track_infos,
    )

    save_index(index_dir=index_dir, meta=meta, intro_embs=intro_arr, outro_embs=outro_arr)

    if errors:
        print(f"[WARN] {len(errors)}개 파일은 처리 실패(인덱스에서 제외). 예: {errors[0][0]} -> {errors[0][1]}", file=sys.stderr)


# -------------------------
# Recommendation
# -------------------------
def _search_candidates(
    query_emb: np.ndarray,
    intro_embs: np.ndarray,
    faiss_index: Optional[object],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, similarities) top-k. Similarity is cosine (inner product of normalized vectors)."""
    query = query_emb.reshape(1, -1).astype(np.float32)
    if faiss_index is not None:
        D, I = faiss_index.search(query, k)
        return I.reshape(-1), D.reshape(-1)

    sims = intro_embs @ query_emb.astype(np.float32)
    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, kth=k - 1)[:k]
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
    다음 곡 추천:
    - 후보: CLAP(outro)->CLAP(intro) 유사도 topK
    - 필터: BPM/Key/History
    - 재랭킹: embed + energy + low_end + mix + key + hype + diversity
    """
    meta, intro_embs, faiss_index = load_index(index_dir)
    embedder = ClapEmbedder(model_name=meta.model_name)

    # Analyze current track
    y_full, sr_full = load_audio(current_path, target_sr=None, max_seconds=None)
    duration_s = float(len(y_full) / sr_full)

    outro_start = max(0.0, duration_s - meta.outro_seconds)
    outro_seg = segment_audio(y_full, sr_full, outro_start, meta.outro_seconds)
    q_emb = embedder.embed_audio(outro_seg, sr_full)

    # analysis @ 22.05k
    if sr_full != 22050:
        y_an = resample_audio(y_full, orig_sr=sr_full, target_sr=22050)
        sr_an = 22050
    else:
        y_an = y_full
        sr_an = sr_full

    y_global = y_an if duration_s <= 120.0 else segment_audio(y_an, sr_an, 0.0, 120.0)

    cur_bpm = estimate_bpm(y_global, sr_an)
    cur_root, cur_mode, cur_clarity = estimate_key(y_global, sr_an)
    cur_energy = estimate_energy(y_global, sr_an)["energy_raw"]
    cur_low = low_end_profile(y_global, sr_an)

    # NEW current hype/diversity features
    cur_onset = estimate_onset_density(y_global, sr_an)
    cur_bright = estimate_brightness(y_global, sr_an)
    cur_vocal = estimate_vocal_presence(y_global, sr_an)
    cur_perc = estimate_percussive_ratio_global(y_global, sr_an)

    cur_ez = (cur_energy - meta.energy_mean) / (meta.energy_std + EPS)
    cur_onz = (cur_onset - meta.onset_mean) / (meta.onset_std + EPS)
    cur_prz = (cur_perc - meta.perc_mean) / (meta.perc_std + EPS)

    # Candidate search
    candidate_k = int(max(top_k, candidate_k))
    idxs, sims = _search_candidates(q_emb, intro_embs, faiss_index, k=min(candidate_k, len(meta.tracks)))

    history = set(history_ids or [])

    # diversity prep from recent history
    id_to_idx = {t.track_id: idx for idx, t in enumerate(meta.tracks)}
    recent_ids = list(history_ids or [])[-8:]
    recent_idxs = [id_to_idx[x] for x in recent_ids if x in id_to_idx]
    recent_embs = intro_embs[recent_idxs] if len(recent_idxs) > 0 else None
    recent_vocals = [meta.tracks[i].vocal_presence for i in recent_idxs] if recent_idxs else []
    recent_brights = [meta.tracks[i].brightness for i in recent_idxs] if recent_idxs else []
    recent_subs = [meta.tracks[i].sub for i in recent_idxs] if recent_idxs else []
    recent_basses = [meta.tracks[i].bass for i in recent_idxs] if recent_idxs else []

    # Weights (tunable)
    w_embed = 0.85
    w_energy = 0.35
    w_low = 0.25
    w_mix = 0.20
    w_key = 0.25
    w_hype = 0.35
    w_div = 0.30

    results: List[Tuple[float, Dict[str, object]]] = []

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
        key_score = 0.0
        if require_key and (cur_clarity >= min_tonal_clarity) and (t.tonal_clarity >= min_tonal_clarity):
            key_score = key_compatibility(cur_root, cur_mode, t.key_root, t.key_mode)
            if key_score <= 0.0:
                continue

        # Soft components
        embed_score = float(sim)

        # Energy target scoring
        t_ez = (t.energy_raw - meta.energy_mean) / (meta.energy_std + EPS)
        if goal == "maintain":
            energy_score = float(np.exp(-abs(t_ez - cur_ez)))
        elif goal == "up":
            delta = float(t_ez - cur_ez)
            energy_score = float(np.exp(-abs(delta - 0.5)))
        elif goal == "down":
            delta = float(t_ez - cur_ez)
            energy_score = float(np.exp(-abs(delta + 0.5)))
        elif goal == "peak":
            delta = float(t_ez - cur_ez)
            energy_score = float(np.exp(-abs(delta - 1.2)))
        else:
            energy_score = float(np.exp(-abs(t_ez - cur_ez)))

        # Low-end compatibility (reduce collision risk)
        low_dist = abs(t.sub - cur_low["sub"]) + 0.7 * abs(t.bass - cur_low["bass"]) + 0.4 * abs(t.lowmid - cur_low["lowmid"])
        low_score = float(np.exp(-3.0 * low_dist))

        # Mixability: candidate intro should be mix-friendly
        mix_score = float(np.clip(t.intro_mix, 0.0, 1.0))

        # Key score usage
        if require_key:
            if (cur_clarity >= min_tonal_clarity) and (t.tonal_clarity >= min_tonal_clarity):
                kscore = float(key_score)
            else:
                kscore = 0.5
        else:
            kscore = 0.0

        # -------------------------
        # Hype score (흥)
        # -------------------------
        t_onz = (t.onset_density - meta.onset_mean) / (meta.onset_std + EPS)
        t_prz = (t.perc_ratio_global - meta.perc_mean) / (meta.perc_std + EPS)

        if goal in ("up", "peak"):
            drive = float(1.0 / (1.0 + np.exp(-(t_onz - cur_onz))))
            perc_boost = float(1.0 / (1.0 + np.exp(-(t_prz - cur_prz))))
        elif goal == "down":
            drive = float(1.0 / (1.0 + np.exp(-(cur_onz - t_onz))))
            perc_boost = float(1.0 / (1.0 + np.exp(-(cur_prz - t_prz))))
        else:
            drive = float(np.exp(-abs(t_onz - cur_onz)))
            perc_boost = float(np.exp(-abs(t_prz - cur_prz)))

        hype_score = float(
            0.45 * energy_score
            + 0.30 * drive
            + 0.20 * perc_boost
            + 0.05 * low_score
        )

        # -------------------------
        # Diversity score (다양성)
        # -------------------------
        # 1) 임베딩 반복 방지: 최근과 max cosine sim가 너무 크면 패널티
        if recent_embs is not None and recent_embs.shape[0] > 0:
            recent_sims = recent_embs @ intro_embs[int(i)]
            max_sim = float(np.max(recent_sims))
            # 0.75 이하 = 충분히 다양(1.0), 0.92 이상 = 매우 비슷(0.0)
            embed_novelty = float(np.clip(1.0 - (max_sim - 0.75) / (0.92 - 0.75 + EPS), 0.0, 1.0))
        else:
            embed_novelty = 0.7

        # 2) 보컬 토글(피로 방지): 최근 평균 대비 '적당한' 차이를 선호(너무 큰 점프는 산만)
        if len(recent_vocals) > 0:
            recent_v = float(np.mean(recent_vocals))
            dv = abs(t.vocal_presence - recent_v)
            vocal_toggle = float(np.exp(-((dv - 0.15) ** 2) / (2 * 0.12**2)))
        else:
            vocal_toggle = 0.5

        # 3) 밝기 변주: 완만한 변화(약 0.08)를 선호
        if len(recent_brights) > 0:
            recent_b = float(np.mean(recent_brights))
            db = abs(t.brightness - recent_b)
            bright_variety = float(np.exp(-((db - 0.08) ** 2) / (2 * 0.08**2)))
        else:
            bright_variety = 0.5

        # 4) 저역 변주: 완만한 변화(약 0.04)를 선호
        if len(recent_subs) > 0:
            recent_sub = float(np.mean(recent_subs))
            recent_bass = float(np.mean(recent_basses))
            dl = abs(t.sub - recent_sub) + 0.7 * abs(t.bass - recent_bass)
            low_variety = float(np.exp(-((dl - 0.04) ** 2) / (2 * 0.05**2)))
        else:
            low_variety = 0.5

        div_score = float(
            0.55 * embed_novelty
            + 0.20 * vocal_toggle
            + 0.15 * bright_variety
            + 0.10 * low_variety
        )

        total = (
            w_embed * embed_score
            + w_energy * energy_score
            + w_low * low_score
            + w_mix * mix_score
            + (w_key * kscore if require_key else 0.0)
            + w_hype * hype_score
            + w_div * div_score
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
                "hype": float(hype_score),
                "diversity": float(div_score),
            },
            "features": {
                "bpm": float(t.bpm),
                "key": f"{_KEY_NAMES_SHARP[t.key_root]} {t.key_mode}",
                "tonal_clarity": float(t.tonal_clarity),
                "energy_z": float(t_ez),
                "sub": float(t.sub),
                "bass": float(t.bass),
                "intro_mix": float(t.intro_mix),
                "onset_density": float(t.onset_density),
                "brightness": float(t.brightness),
                "vocal_presence": float(t.vocal_presence),
                "perc_ratio_global": float(t.perc_ratio_global),
            },
            "rank_in_candidates": int(rank_pos),
        }
        results.append((float(total), detail))

    results.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in results[: int(top_k)]]


# -------------------------
# Continuous session helper
# -------------------------
class SessionRecommender:
    """Keeps history and provides continuous recommendations."""

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

    def next(self, current_path: str, goal: str = "maintain", top_k: int = 10, candidate_k: int = 200) -> List[Dict[str, object]]:
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


# -------------------------
# CLI
# -------------------------
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
    p = argparse.ArgumentParser(description="Raw-audio DJ-style next-track recommender (CLAP + constraints + hype/diversity)")
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
    r.add_argument("--candidate_k", type=int, default=200, help="Initial candidate size before filtering/rerank")
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
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        if os.environ.get("DJREC_DEBUG", "").strip() == "1":
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()