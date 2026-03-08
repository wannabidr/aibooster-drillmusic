/// Crossfade curve type for volume transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadeCurve {
    Linear,
    EqualPower,
}

impl CrossfadeCurve {
    /// Calculate gain values for tracks A and B at a given position.
    /// `position` ranges from 0.0 (100% A) to 1.0 (100% B).
    /// Returns (gain_a, gain_b).
    pub fn gain_at(&self, position: f64) -> (f64, f64) {
        let position = position.clamp(0.0, 1.0);
        match self {
            CrossfadeCurve::Linear => (1.0 - position, position),
            CrossfadeCurve::EqualPower => {
                let angle = position * std::f64::consts::FRAC_PI_2;
                (angle.cos(), angle.sin())
            }
        }
    }
}

/// High-level transition style that determines the overall mixing approach.
/// Phase 1 uses volume-only crossfades. Phase 2 adds EQ/filter automation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CrossfadeStyle {
    /// Simple volume crossfade using the specified curve (Phase 1).
    VolumeFade(CrossfadeCurve),
    // Phase 2 variants (uncomment when implementing):
    // /// Swap bass EQ between tracks at the transition midpoint.
    // EQSwap,
    // /// Gradual low-pass filter sweep on outgoing track.
    // FilterSweep,
    // /// Echo/delay tail on outgoing track.
    // EchoOut,
}

impl CrossfadeStyle {
    /// Returns the underlying crossfade curve for gain calculations.
    /// Phase 2 styles will have their own default curves.
    pub fn curve(&self) -> CrossfadeCurve {
        match self {
            CrossfadeStyle::VolumeFade(curve) => *curve,
        }
    }
}

/// Configuration for a crossfade transition between two tracks.
#[derive(Debug, Clone)]
pub struct TransitionParams {
    transition_beats: u32,
    style: CrossfadeStyle,
    mix_point_a: u64,
    mix_point_b: u64,
}

impl TransitionParams {
    pub fn new(
        transition_beats: u32,
        style: CrossfadeStyle,
        mix_point_a: u64,
        mix_point_b: u64,
    ) -> Result<Self, TransitionError> {
        if transition_beats == 0 {
            return Err(TransitionError::InvalidTransitionBeats);
        }
        Ok(Self {
            transition_beats,
            style,
            mix_point_a,
            mix_point_b,
        })
    }

    /// Create a 16-beat linear crossfade.
    pub fn default_16_beat(mix_point_a: u64, mix_point_b: u64) -> Self {
        Self {
            transition_beats: 16,
            style: CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear),
            mix_point_a,
            mix_point_b,
        }
    }

    /// Create a 32-beat equal-power crossfade.
    pub fn default_32_beat(mix_point_a: u64, mix_point_b: u64) -> Self {
        Self {
            transition_beats: 32,
            style: CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower),
            mix_point_a,
            mix_point_b,
        }
    }

    pub fn transition_beats(&self) -> u32 {
        self.transition_beats
    }

    pub fn style(&self) -> CrossfadeStyle {
        self.style
    }

    /// Convenience: returns the crossfade curve from the style.
    pub fn curve_type(&self) -> CrossfadeCurve {
        self.style.curve()
    }

    pub fn mix_point_a(&self) -> u64 {
        self.mix_point_a
    }

    pub fn mix_point_b(&self) -> u64 {
        self.mix_point_b
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TransitionError {
    InvalidTransitionBeats,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransitionBeats => write!(f, "transition beats must be > 0"),
        }
    }
}

impl std::error::Error for TransitionError {}

#[cfg(test)]
mod tests {
    use super::*;

    // -- CrossfadeCurve tests --

    #[test]
    fn linear_at_start() {
        let (a, b) = CrossfadeCurve::Linear.gain_at(0.0);
        assert!((a - 1.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);
    }

    #[test]
    fn linear_at_end() {
        let (a, b) = CrossfadeCurve::Linear.gain_at(1.0);
        assert!((a - 0.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
    }

    #[test]
    fn linear_at_midpoint() {
        let (a, b) = CrossfadeCurve::Linear.gain_at(0.5);
        assert!((a - 0.5).abs() < 1e-10);
        assert!((b - 0.5).abs() < 1e-10);
    }

    #[test]
    fn equal_power_at_start() {
        let (a, b) = CrossfadeCurve::EqualPower.gain_at(0.0);
        assert!((a - 1.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);
    }

    #[test]
    fn equal_power_at_end() {
        let (a, b) = CrossfadeCurve::EqualPower.gain_at(1.0);
        assert!((a - 0.0).abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equal_power_at_midpoint_preserves_energy() {
        let (a, b) = CrossfadeCurve::EqualPower.gain_at(0.5);
        // At midpoint, both gains should be ~0.707 (1/sqrt(2))
        let expected = 1.0_f64 / 2.0_f64.sqrt();
        assert!((a - expected).abs() < 1e-6);
        assert!((b - expected).abs() < 1e-6);
        // Energy preservation: a^2 + b^2 = 1
        assert!((a * a + b * b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn equal_power_energy_preserved_throughout() {
        for i in 0..=100 {
            let pos = i as f64 / 100.0;
            let (a, b) = CrossfadeCurve::EqualPower.gain_at(pos);
            assert!(
                (a * a + b * b - 1.0).abs() < 1e-6,
                "energy not preserved at position {pos}: a={a}, b={b}"
            );
        }
    }

    #[test]
    fn gain_clamps_below_zero() {
        let (a, b) = CrossfadeCurve::Linear.gain_at(-0.5);
        assert!((a - 1.0).abs() < 1e-10);
        assert!((b - 0.0).abs() < 1e-10);
    }

    #[test]
    fn gain_clamps_above_one() {
        let (a, b) = CrossfadeCurve::Linear.gain_at(1.5);
        assert!((a - 0.0).abs() < 1e-10);
        assert!((b - 1.0).abs() < 1e-10);
    }

    // -- CrossfadeStyle tests --

    #[test]
    fn style_volume_fade_returns_curve() {
        let style = CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower);
        assert_eq!(style.curve(), CrossfadeCurve::EqualPower);
    }

    #[test]
    fn style_volume_fade_linear() {
        let style = CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear);
        assert_eq!(style.curve(), CrossfadeCurve::Linear);
    }

    // -- TransitionParams tests --

    #[test]
    fn create_valid_transition() {
        let style = CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear);
        let params = TransitionParams::new(16, style, 1000, 2000).unwrap();
        assert_eq!(params.transition_beats(), 16);
        assert_eq!(params.style(), style);
        assert_eq!(params.curve_type(), CrossfadeCurve::Linear);
        assert_eq!(params.mix_point_a(), 1000);
        assert_eq!(params.mix_point_b(), 2000);
    }

    #[test]
    fn reject_zero_transition_beats() {
        let style = CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear);
        assert!(TransitionParams::new(0, style, 0, 0).is_err());
    }

    #[test]
    fn default_16_beat() {
        let params = TransitionParams::default_16_beat(100, 200);
        assert_eq!(params.transition_beats(), 16);
        assert_eq!(params.curve_type(), CrossfadeCurve::Linear);
        assert_eq!(params.style(), CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear));
    }

    #[test]
    fn default_32_beat() {
        let params = TransitionParams::default_32_beat(100, 200);
        assert_eq!(params.transition_beats(), 32);
        assert_eq!(params.curve_type(), CrossfadeCurve::EqualPower);
        assert_eq!(params.style(), CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower));
    }
}
