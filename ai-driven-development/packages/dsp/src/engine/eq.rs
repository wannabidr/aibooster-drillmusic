//! 3-band EQ automation engine for DJ transitions.
//!
//! Splits audio into low/mid/high frequency bands and applies
//! per-band gain automation for smooth EQ-based transitions.

use crate::domain::blend_params::{AutomationCurve, BandCurves, EqAutomation};

/// Crossover frequencies for 3-band EQ split.
const LOW_MID_CROSSOVER: f64 = 250.0;
const MID_HIGH_CROSSOVER: f64 = 4000.0;

/// Simple biquad filter state for EQ band splitting.
#[derive(Debug, Clone)]
struct BiquadState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadState {
    fn new() -> Self {
        Self { x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    fn process(&mut self, input: f64, coeffs: &BiquadCoeffs) -> f64 {
        let output = coeffs.b0 * input + coeffs.b1 * self.x1 + coeffs.b2 * self.x2
            - coeffs.a1 * self.y1 - coeffs.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }
}

#[derive(Debug, Clone)]
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl BiquadCoeffs {
    /// 2nd-order Butterworth low-pass filter.
    fn lowpass(cutoff: f64, sample_rate: f64) -> Self {
        let omega = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = std::f64::consts::FRAC_1_SQRT_2;
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
        }
    }

    /// 2nd-order Butterworth high-pass filter.
    fn highpass(cutoff: f64, sample_rate: f64) -> Self {
        let omega = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let q = std::f64::consts::FRAC_1_SQRT_2;
        let alpha = sin_omega / (2.0 * q);

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
        }
    }
}

/// Process state for one channel of 3-band EQ splitting.
#[derive(Debug, Clone)]
struct ChannelEqState {
    lp_low: BiquadState,
    hp_mid: BiquadState,
    lp_mid: BiquadState,
    hp_high: BiquadState,
}

impl ChannelEqState {
    fn new() -> Self {
        Self {
            lp_low: BiquadState::new(),
            hp_mid: BiquadState::new(),
            lp_mid: BiquadState::new(),
            hp_high: BiquadState::new(),
        }
    }
}

/// 3-band EQ processor.
pub struct EqEngine;

impl EqEngine {
    /// Apply EQ automation to a mono audio buffer in-place.
    ///
    /// `samples` - mono audio samples
    /// `sample_rate` - sample rate in Hz
    /// `band_curves` - gain curves for low/mid/high bands
    /// `total_frames` - number of frames this automation spans (for position calculation)
    /// `frame_offset` - starting frame offset within the automation range
    ///
    /// Returns the processed samples.
    pub fn apply_eq_automation(
        samples: &[f32],
        sample_rate: u32,
        band_curves: &BandCurves,
        total_frames: usize,
        frame_offset: usize,
    ) -> Vec<f32> {
        let sr = sample_rate as f64;
        let lp_low_coeffs = BiquadCoeffs::lowpass(LOW_MID_CROSSOVER, sr);
        let hp_mid_coeffs = BiquadCoeffs::highpass(LOW_MID_CROSSOVER, sr);
        let lp_mid_coeffs = BiquadCoeffs::lowpass(MID_HIGH_CROSSOVER, sr);
        let hp_high_coeffs = BiquadCoeffs::highpass(MID_HIGH_CROSSOVER, sr);

        let mut state = ChannelEqState::new();
        let mut output = Vec::with_capacity(samples.len());

        for (i, &sample) in samples.iter().enumerate() {
            let position = if total_frames <= 1 {
                1.0
            } else {
                (frame_offset + i) as f64 / (total_frames - 1) as f64
            };

            let input = sample as f64;

            // Split into 3 bands
            let low = state.lp_low.process(input, &lp_low_coeffs);
            let mid_hp = state.hp_mid.process(input, &hp_mid_coeffs);
            let mid = state.lp_mid.process(mid_hp, &lp_mid_coeffs);
            let high = state.hp_high.process(input, &hp_high_coeffs);

            // Apply per-band gain automation
            let gain_low = band_curves.low.value_at(position);
            let gain_mid = band_curves.mid.value_at(position);
            let gain_high = band_curves.high.value_at(position);

            let result = low * gain_low + mid * gain_mid + high * gain_high;
            output.push(result as f32);
        }

        output
    }

    /// Create EQ automation for a "low-swap" transition.
    ///
    /// Track A lows fade out while track B lows fade in at the midpoint.
    /// Mids and highs follow a standard crossfade curve.
    pub fn low_swap_automation() -> EqAutomation {
        EqAutomation {
            track_a_curves: BandCurves {
                low: AutomationCurve::new(vec![
                    (0.0, 1.0),
                    (0.45, 1.0),
                    (0.5, 0.0),
                    (1.0, 0.0),
                ]).unwrap(),
                mid: AutomationCurve::ramp(1.0, 0.0),
                high: AutomationCurve::ramp(1.0, 0.0),
            },
            track_b_curves: BandCurves {
                low: AutomationCurve::new(vec![
                    (0.0, 0.0),
                    (0.5, 0.0),
                    (0.55, 1.0),
                    (1.0, 1.0),
                ]).unwrap(),
                mid: AutomationCurve::ramp(0.0, 1.0),
                high: AutomationCurve::ramp(0.0, 1.0),
            },
        }
    }

    /// Create EQ automation for a "high-cut" transition.
    ///
    /// Track A highs cut early, mids follow, lows last.
    /// Track B builds in reverse order.
    pub fn high_cut_automation() -> EqAutomation {
        EqAutomation {
            track_a_curves: BandCurves {
                high: AutomationCurve::new(vec![
                    (0.0, 1.0),
                    (0.3, 0.0),
                    (1.0, 0.0),
                ]).unwrap(),
                mid: AutomationCurve::new(vec![
                    (0.0, 1.0),
                    (0.5, 0.0),
                    (1.0, 0.0),
                ]).unwrap(),
                low: AutomationCurve::new(vec![
                    (0.0, 1.0),
                    (0.7, 1.0),
                    (1.0, 0.0),
                ]).unwrap(),
            },
            track_b_curves: BandCurves {
                low: AutomationCurve::new(vec![
                    (0.0, 0.0),
                    (0.3, 1.0),
                    (1.0, 1.0),
                ]).unwrap(),
                mid: AutomationCurve::new(vec![
                    (0.0, 0.0),
                    (0.5, 1.0),
                    (1.0, 1.0),
                ]).unwrap(),
                high: AutomationCurve::new(vec![
                    (0.0, 0.0),
                    (0.7, 0.0),
                    (1.0, 1.0),
                ]).unwrap(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::blend_params::AutomationCurve;

    /// Generate a sine wave for testing.
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
    fn unity_gain_preserves_signal() {
        let sr = 44100;
        let samples = sine_wave(440.0, sr, 0.5);
        let unity_curves = BandCurves {
            low: AutomationCurve::constant(1.0),
            mid: AutomationCurve::constant(1.0),
            high: AutomationCurve::constant(1.0),
        };

        let output = EqEngine::apply_eq_automation(
            &samples, sr, &unity_curves, samples.len(), 0,
        );

        assert_eq!(output.len(), samples.len());
        // After the filter settles, signal should be roughly preserved.
        // 3-band splitting is imperfect (phase and overlap), so allow wider tolerance.
        let input_rms = rms(&samples[5000..]);
        let output_rms = rms(&output[5000..]);
        let ratio = output_rms / input_rms;
        assert!(
            ratio > 0.5 && ratio < 1.8,
            "unity gain should roughly preserve signal, ratio: {ratio}"
        );
    }

    #[test]
    fn zero_gain_silences_signal() {
        let sr = 44100;
        let samples = sine_wave(440.0, sr, 0.1);
        let zero_curves = BandCurves {
            low: AutomationCurve::constant(0.0),
            mid: AutomationCurve::constant(0.0),
            high: AutomationCurve::constant(0.0),
        };

        let output = EqEngine::apply_eq_automation(
            &samples, sr, &zero_curves, samples.len(), 0,
        );

        let output_rms = rms(&output[200..]);
        assert!(
            output_rms < 0.01,
            "zero gain should silence signal, rms: {output_rms}"
        );
    }

    #[test]
    fn low_band_affects_bass_frequencies() {
        let sr = 44100;
        // 80 Hz bass tone
        let bass = sine_wave(80.0, sr, 0.5);
        let cut_low = BandCurves {
            low: AutomationCurve::constant(0.0),
            mid: AutomationCurve::constant(1.0),
            high: AutomationCurve::constant(1.0),
        };

        let output = EqEngine::apply_eq_automation(
            &bass, sr, &cut_low, bass.len(), 0,
        );

        let input_rms = rms(&bass[2000..]);
        let output_rms = rms(&output[2000..]);
        assert!(
            output_rms < input_rms * 0.3,
            "cutting low band should reduce bass: input_rms={input_rms}, output_rms={output_rms}"
        );
    }

    #[test]
    fn high_band_affects_treble_frequencies() {
        let sr = 44100;
        // 8000 Hz treble tone
        let treble = sine_wave(8000.0, sr, 0.5);
        let cut_high = BandCurves {
            low: AutomationCurve::constant(1.0),
            mid: AutomationCurve::constant(1.0),
            high: AutomationCurve::constant(0.0),
        };

        let output = EqEngine::apply_eq_automation(
            &treble, sr, &cut_high, treble.len(), 0,
        );

        let input_rms = rms(&treble[2000..]);
        let output_rms = rms(&output[2000..]);
        assert!(
            output_rms < input_rms * 0.3,
            "cutting high band should reduce treble: input_rms={input_rms}, output_rms={output_rms}"
        );
    }

    #[test]
    fn low_swap_automation_structure() {
        let auto = EqEngine::low_swap_automation();

        // Track A low: unity at start, zero at end
        assert!((auto.track_a_curves.low.value_at(0.0) - 1.0).abs() < 1e-10);
        assert!((auto.track_a_curves.low.value_at(1.0) - 0.0).abs() < 1e-10);

        // Track B low: zero at start, unity at end
        assert!((auto.track_b_curves.low.value_at(0.0) - 0.0).abs() < 1e-10);
        assert!((auto.track_b_curves.low.value_at(1.0) - 1.0).abs() < 1e-10);

        // At the swap point (~0.5), track A lows should be fading out
        assert!(auto.track_a_curves.low.value_at(0.5) < 0.1);
        // And track B lows should be starting to come in
        assert!(auto.track_b_curves.low.value_at(0.55) > 0.9);
    }

    #[test]
    fn high_cut_automation_structure() {
        let auto = EqEngine::high_cut_automation();

        // Track A highs cut early (by 0.3)
        assert!((auto.track_a_curves.high.value_at(0.3) - 0.0).abs() < 1e-10);
        // Track A mids follow (by 0.5)
        assert!((auto.track_a_curves.mid.value_at(0.5) - 0.0).abs() < 1e-10);
        // Track A lows last (still unity at 0.5)
        assert!((auto.track_a_curves.low.value_at(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eq_automation_produces_no_clicks() {
        let sr = 44100;
        let samples = sine_wave(440.0, sr, 1.0);
        let curves = BandCurves {
            low: AutomationCurve::ramp(1.0, 0.0),
            mid: AutomationCurve::ramp(1.0, 0.0),
            high: AutomationCurve::ramp(1.0, 0.0),
        };

        let output = EqEngine::apply_eq_automation(
            &samples, sr, &curves, samples.len(), 0,
        );

        // Check for clicks: no adjacent sample should differ by more than 0.1
        // (skip first few samples for filter settling)
        for i in 201..output.len() {
            let diff = (output[i] - output[i - 1]).abs();
            assert!(
                diff < 0.1,
                "click detected at sample {i}: diff={diff}, prev={}, curr={}",
                output[i - 1], output[i]
            );
        }
    }
}
