//! Filter sweep engine for DJ transitions.
//!
//! Provides HPF/LPF sweeps with resonance control and smooth automation.

use crate::domain::blend_params::{AutomationCurve, FilterSweepParams, FilterType};

/// State-variable filter for smooth sweeps with resonance.
/// Uses the Chamberlin SVF topology for numerical stability.
#[derive(Debug, Clone)]
struct SvfState {
    low: f64,
    band: f64,
    high: f64,
}

impl SvfState {
    fn new() -> Self {
        Self { low: 0.0, band: 0.0, high: 0.0 }
    }

    /// Process one sample through the state-variable filter.
    /// Returns (low, band, high) outputs.
    /// Uses two iterations of the Chamberlin SVF for better stability.
    fn process(&mut self, input: f64, f_coeff: f64, q_coeff: f64) -> (f64, f64, f64) {
        // Clamp f_coeff for stability (must be < 1.0 for Chamberlin SVF)
        let f = f_coeff.min(0.99);
        // Two iterations for oversampled accuracy
        for _ in 0..2 {
            self.high = input - self.low - q_coeff * self.band;
            self.band += f * self.high;
            self.low += f * self.band;
        }
        // Clamp outputs to prevent runaway values
        self.low = self.low.clamp(-10.0, 10.0);
        self.band = self.band.clamp(-10.0, 10.0);
        self.high = self.high.clamp(-10.0, 10.0);
        (self.low, self.band, self.high)
    }
}

/// Filter sweep processor.
pub struct FilterEngine;

impl FilterEngine {
    /// Apply a filter sweep to mono audio samples.
    ///
    /// The cutoff frequency is automated over the duration of the buffer.
    ///
    /// # Arguments
    /// * `samples` - mono audio input
    /// * `sample_rate` - sample rate in Hz
    /// * `params` - filter type, cutoff curve, and resonance
    /// * `total_frames` - total frames in the automation range
    /// * `frame_offset` - starting frame offset
    pub fn apply_sweep(
        samples: &[f32],
        sample_rate: u32,
        params: &FilterSweepParams,
        total_frames: usize,
        frame_offset: usize,
    ) -> Vec<f32> {
        let sr = sample_rate as f64;
        let mut state = SvfState::new();
        let mut output = Vec::with_capacity(samples.len());

        // Damping factor from resonance (Q)
        let q_coeff = 1.0 / params.resonance;

        for (i, &sample) in samples.iter().enumerate() {
            let position = if total_frames <= 1 {
                1.0
            } else {
                (frame_offset + i) as f64 / (total_frames - 1) as f64
            };

            let cutoff = params.cutoff_curve.value_at(position);
            // Convert cutoff frequency to filter coefficient
            // Clamp to prevent instability at Nyquist
            let max_freq = sr * 0.45;
            let clamped_cutoff = cutoff.clamp(20.0, max_freq);
            let f_coeff = 2.0 * (std::f64::consts::PI * clamped_cutoff / sr).sin();

            let (low, _band, high) = state.process(sample as f64, f_coeff, q_coeff);

            let out = match params.filter_type {
                FilterType::LowPass => low,
                FilterType::HighPass => high,
            };
            output.push(out as f32);
        }

        output
    }

    /// Create a standard HPF sweep for an outgoing track:
    /// starts fully open (20 Hz cutoff) and sweeps up to remove bass/mids.
    pub fn hpf_sweep_out(resonance: f64) -> FilterSweepParams {
        FilterSweepParams {
            filter_type: FilterType::HighPass,
            cutoff_curve: AutomationCurve::new(vec![
                (0.0, 20.0),
                (0.7, 2000.0),
                (1.0, 10000.0),
            ]).unwrap(),
            resonance,
        }
    }

    /// Create a standard LPF sweep for an incoming track:
    /// starts with heavy filtering and opens up fully.
    pub fn lpf_sweep_in(resonance: f64) -> FilterSweepParams {
        FilterSweepParams {
            filter_type: FilterType::LowPass,
            cutoff_curve: AutomationCurve::new(vec![
                (0.0, 200.0),
                (0.3, 1000.0),
                (1.0, 20000.0),
            ]).unwrap(),
            resonance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, sample_rate: u32, duration_secs: f64) -> Vec<f32> {
        let num_samples = (sample_rate as f64 * duration_secs) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32
            })
            .collect()
    }

    fn rms(samples: &[f32]) -> f64 {
        let sum: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        (sum / samples.len() as f64).sqrt()
    }

    #[test]
    fn lowpass_attenuates_high_frequencies() {
        let sr = 44100;
        let treble = sine_wave(8000.0, sr, 0.5);
        let params = FilterSweepParams {
            filter_type: FilterType::LowPass,
            cutoff_curve: AutomationCurve::constant(500.0),
            resonance: 0.707,
        };

        let output = FilterEngine::apply_sweep(&treble, sr, &params, treble.len(), 0);

        let input_rms = rms(&treble[2000..]);
        let output_rms = rms(&output[2000..]);
        assert!(
            output_rms < input_rms * 0.15,
            "LPF at 500Hz should attenuate 8kHz signal: ratio={}",
            output_rms / input_rms
        );
    }

    #[test]
    fn highpass_attenuates_low_frequencies() {
        let sr = 44100;
        let bass = sine_wave(80.0, sr, 0.5);
        let params = FilterSweepParams {
            filter_type: FilterType::HighPass,
            cutoff_curve: AutomationCurve::constant(2000.0),
            resonance: 0.707,
        };

        let output = FilterEngine::apply_sweep(&bass, sr, &params, bass.len(), 0);

        let input_rms = rms(&bass[2000..]);
        let output_rms = rms(&output[2000..]);
        assert!(
            output_rms < input_rms * 0.15,
            "HPF at 2kHz should attenuate 80Hz signal: ratio={}",
            output_rms / input_rms
        );
    }

    #[test]
    fn lowpass_passes_low_frequencies() {
        let sr = 44100;
        let bass = sine_wave(100.0, sr, 0.5);
        let params = FilterSweepParams {
            filter_type: FilterType::LowPass,
            cutoff_curve: AutomationCurve::constant(5000.0),
            resonance: 0.707,
        };

        let output = FilterEngine::apply_sweep(&bass, sr, &params, bass.len(), 0);

        let input_rms = rms(&bass[2000..]);
        let output_rms = rms(&output[2000..]);
        let ratio = output_rms / input_rms;
        assert!(
            ratio > 0.8,
            "LPF at 5kHz should pass 100Hz signal: ratio={ratio}"
        );
    }

    #[test]
    fn hpf_sweep_removes_bass_at_end() {
        let sr = 44100;
        let bass = sine_wave(100.0, sr, 2.0);
        let params = FilterEngine::hpf_sweep_out(1.0);

        let output = FilterEngine::apply_sweep(&bass, sr, &params, bass.len(), 0);

        // Early in the sweep (position ~0.1), bass should be present (HPF cutoff ~20Hz)
        let early_start = (bass.len() as f64 * 0.05) as usize;
        let early_end = (bass.len() as f64 * 0.15) as usize;
        let early_rms = rms(&output[early_start..early_end]);
        // Late in the sweep (position ~0.9), bass should be attenuated (HPF cutoff ~8kHz)
        let late_start = (bass.len() as f64 * 0.85) as usize;
        let late_end = (bass.len() as f64 * 0.95) as usize;
        let late_rms = rms(&output[late_start..late_end]);
        assert!(
            late_rms < early_rms,
            "HPF sweep should attenuate bass at end: early={early_rms}, late={late_rms}"
        );
    }

    #[test]
    fn lpf_sweep_opens_up_treble() {
        let sr = 44100;
        let treble = sine_wave(5000.0, sr, 2.0);
        let params = FilterEngine::lpf_sweep_in(1.0);

        let output = FilterEngine::apply_sweep(&treble, sr, &params, treble.len(), 0);

        // Early in the sweep (cutoff ~200Hz), treble should be attenuated
        let early_start = (treble.len() as f64 * 0.05) as usize;
        let early_end = (treble.len() as f64 * 0.15) as usize;
        let early_rms = rms(&output[early_start..early_end]);
        // Late in the sweep (cutoff ~20kHz), treble should pass through
        let late_start = (treble.len() as f64 * 0.85) as usize;
        let late_end = (treble.len() as f64 * 0.95) as usize;
        let late_rms = rms(&output[late_start..late_end]);
        assert!(
            late_rms > early_rms,
            "LPF sweep should open up treble: early={early_rms}, late={late_rms}"
        );
    }

    #[test]
    fn filter_sweep_produces_no_clicks() {
        let sr = 44100;
        let samples = sine_wave(440.0, sr, 1.0);
        let params = FilterEngine::hpf_sweep_out(1.0);

        let output = FilterEngine::apply_sweep(&samples, sr, &params, samples.len(), 0);

        // No adjacent samples should differ by more than 0.3 (smooth automation)
        for i in 201..output.len() {
            let diff = (output[i] - output[i - 1]).abs();
            assert!(
                diff < 0.3,
                "click detected at sample {i}: diff={diff}"
            );
        }
    }

    #[test]
    fn resonance_creates_peak_at_cutoff() {
        let sr = 44100;
        // White noise approximation: many frequencies
        let noise: Vec<f32> = (0..sr as usize)
            .map(|i| {
                let t = i as f64 / sr as f64;
                // Mix of many frequencies
                ((100.0 * t * std::f64::consts::TAU).sin()
                    + (500.0 * t * std::f64::consts::TAU).sin()
                    + (1000.0 * t * std::f64::consts::TAU).sin()
                    + (3000.0 * t * std::f64::consts::TAU).sin()
                    + (8000.0 * t * std::f64::consts::TAU).sin()) as f32
                    / 5.0
            })
            .collect();

        // Low resonance
        let params_low_q = FilterSweepParams {
            filter_type: FilterType::LowPass,
            cutoff_curve: AutomationCurve::constant(1000.0),
            resonance: 0.5,
        };
        let output_low_q = FilterEngine::apply_sweep(&noise, sr, &params_low_q, noise.len(), 0);

        // High resonance
        let params_high_q = FilterSweepParams {
            filter_type: FilterType::LowPass,
            cutoff_curve: AutomationCurve::constant(1000.0),
            resonance: 5.0,
        };
        let output_high_q = FilterEngine::apply_sweep(&noise, sr, &params_high_q, noise.len(), 0);

        // High resonance should have higher peak amplitude near cutoff
        let max_low = output_low_q[5000..].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let max_high = output_high_q[5000..].iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(
            max_high > max_low,
            "higher resonance should create higher peaks: low_q_max={max_low}, high_q_max={max_high}"
        );
    }
}
