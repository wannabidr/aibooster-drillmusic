"""Genre feature extraction from audio signals."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


def extract_genre_features(audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """Extract genre-relevant features from mono audio signal.

    Features (total 42 dims):
    - MFCCs: 20 coefficients (mean across frames)
    - Spectral centroid: 1 (mean)
    - Spectral rolloff: 1 (mean)
    - Spectral bandwidth: 1 (mean)
    - Zero crossing rate: 1 (mean)
    - RMS energy: 1 (mean)
    - Tempo histogram: 8 bins
    - Onset strength stats: 4 (mean, std, max, median)
    - Spectral contrast: 5 bands (mean)

    Returns 1-D float32 array of shape (42,).
    """
    frame_size = 2048
    hop_size = 512

    frames = _frame_audio(audio, frame_size, hop_size)
    if len(frames) == 0:
        return np.zeros(42, dtype=np.float32)

    power_spectra = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    mfccs = _compute_mfccs(power_spectra, sample_rate, frame_size, n_mfcc=20)
    centroid = _spectral_centroid(power_spectra, freqs)
    rolloff = _spectral_rolloff(power_spectra, freqs)
    bandwidth = _spectral_bandwidth(power_spectra, freqs, centroid)
    zcr = _zero_crossing_rate(frames)
    rms = _rms_energy(frames)
    tempo_hist = _tempo_histogram(audio, sample_rate, hop_size)
    onset_stats = _onset_strength_stats(power_spectra)
    contrast = _spectral_contrast(power_spectra, freqs)

    features = np.concatenate(
        [
            mfccs,  # 20
            [centroid],  # 1
            [rolloff],  # 1
            [bandwidth],  # 1
            [zcr],  # 1
            [rms],  # 1
            tempo_hist,  # 8
            onset_stats,  # 4
            contrast,  # 5
        ]
    )
    return features.astype(np.float32)


def _frame_audio(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    n_frames = max(0, 1 + (len(audio) - frame_size) // hop_size)
    if n_frames == 0:
        return np.array([])
    indices = np.arange(frame_size)[None, :] + np.arange(n_frames)[:, None] * hop_size
    return audio[indices] * np.hanning(frame_size)


def _compute_mfccs(
    power_spectra: np.ndarray, sample_rate: int, frame_size: int, n_mfcc: int = 20
) -> np.ndarray:
    n_mels = 40
    mel_filters = _mel_filterbank(n_mels, frame_size, sample_rate)
    mel_spec = power_spectra @ mel_filters.T
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)
    mfcc_frames = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return np.mean(mfcc_frames, axis=0)


def _mel_filterbank(n_mels: int, frame_size: int, sample_rate: int) -> np.ndarray:
    n_fft_bins = frame_size // 2 + 1
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((frame_size + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft_bins))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)
    return filters


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _spectral_centroid(power_spectra: np.ndarray, freqs: np.ndarray) -> float:
    magnitudes = np.sqrt(power_spectra)
    centroid_per_frame = np.sum(freqs * magnitudes, axis=1) / (np.sum(magnitudes, axis=1) + 1e-10)
    return float(np.mean(centroid_per_frame))


def _spectral_rolloff(
    power_spectra: np.ndarray, freqs: np.ndarray, threshold: float = 0.85
) -> float:
    cumsum = np.cumsum(power_spectra, axis=1)
    total = cumsum[:, -1:]
    rolloff_idx = np.argmax(cumsum >= threshold * total, axis=1)
    rolloff_freqs = freqs[rolloff_idx]
    return float(np.mean(rolloff_freqs))


def _spectral_bandwidth(power_spectra: np.ndarray, freqs: np.ndarray, centroid: float) -> float:
    magnitudes = np.sqrt(power_spectra)
    deviation = np.abs(freqs - centroid)
    bw_per_frame = np.sum(deviation * magnitudes, axis=1) / (np.sum(magnitudes, axis=1) + 1e-10)
    return float(np.mean(bw_per_frame))


def _zero_crossing_rate(frames: np.ndarray) -> float:
    signs = np.sign(frames)
    crossings = np.abs(np.diff(signs, axis=1))
    zcr_per_frame = np.sum(crossings > 0, axis=1) / frames.shape[1]
    return float(np.mean(zcr_per_frame))


def _rms_energy(frames: np.ndarray) -> float:
    rms_per_frame = np.sqrt(np.mean(frames**2, axis=1))
    return float(np.mean(rms_per_frame))


def _tempo_histogram(audio: np.ndarray, sample_rate: int, hop_size: int) -> np.ndarray:
    onset = np.abs(np.diff(audio))
    frame_len = sample_rate // 4  # 250ms frames
    n_onset_frames = max(1, len(onset) // frame_len)
    onset_strength = np.zeros(n_onset_frames)
    for i in range(n_onset_frames):
        chunk = onset[i * frame_len : (i + 1) * frame_len]
        onset_strength[i] = np.mean(chunk) if len(chunk) > 0 else 0.0

    if len(onset_strength) < 4:
        return np.zeros(8, dtype=np.float32)

    autocorr = np.correlate(onset_strength, onset_strength, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    hist = np.zeros(8, dtype=np.float32)
    frames_per_sec = sample_rate / frame_len
    for lag in range(1, min(len(autocorr), int(frames_per_sec * 2))):
        bpm = 60.0 * frames_per_sec / lag
        if 60 <= bpm <= 200:
            bin_idx = min(7, int((bpm - 60) / (140 / 8)))
            hist[bin_idx] += autocorr[lag]

    total = np.sum(hist)
    if total > 0:
        hist /= total
    return hist


def _onset_strength_stats(power_spectra: np.ndarray) -> np.ndarray:
    spectral_flux = np.sum(np.maximum(np.diff(power_spectra, axis=0), 0), axis=1)
    if len(spectral_flux) == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [
            np.mean(spectral_flux),
            np.std(spectral_flux),
            np.max(spectral_flux),
            np.median(spectral_flux),
        ],
        dtype=np.float32,
    )


def _spectral_contrast(
    power_spectra: np.ndarray, freqs: np.ndarray, n_bands: int = 5
) -> np.ndarray:
    n_bins = power_spectra.shape[1]
    band_edges = np.logspace(np.log10(1), np.log10(n_bins), n_bands + 1).astype(int)
    band_edges = np.clip(band_edges, 0, n_bins)

    contrast = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        lo, hi = band_edges[i], band_edges[i + 1]
        if hi <= lo:
            continue
        band = power_spectra[:, lo:hi]
        sorted_band = np.sort(band, axis=1)
        n_cols = sorted_band.shape[1]
        alpha = max(1, n_cols // 5)
        peaks = np.mean(sorted_band[:, -alpha:], axis=1)
        valleys = np.mean(sorted_band[:, :alpha], axis=1)
        contrast[i] = float(np.mean(np.log10(peaks + 1e-10) - np.log10(valleys + 1e-10)))

    return contrast
