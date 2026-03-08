/// Type of mix point within a track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixType {
    Intro,
    Outro,
    Breakdown,
    Drop,
}

/// A suitable point in a track for mixing in or out.
#[derive(Debug, Clone)]
pub struct MixPoint {
    sample_position: u64,
    beat_index: usize,
    confidence: f64,
    mix_type: MixType,
}

impl MixPoint {
    pub fn new(
        sample_position: u64,
        beat_index: usize,
        confidence: f64,
        mix_type: MixType,
    ) -> Result<Self, MixPointError> {
        if !(0.0..=1.0).contains(&confidence) || !confidence.is_finite() {
            return Err(MixPointError::InvalidConfidence(confidence));
        }
        Ok(Self {
            sample_position,
            beat_index,
            confidence,
            mix_type,
        })
    }

    pub fn sample_position(&self) -> u64 {
        self.sample_position
    }

    pub fn beat_index(&self) -> usize {
        self.beat_index
    }

    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    pub fn mix_type(&self) -> MixType {
        self.mix_type
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MixPointError {
    InvalidConfidence(f64),
}

impl std::fmt::Display for MixPointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfidence(c) => write!(f, "confidence must be between 0.0 and 1.0, got {c}"),
        }
    }
}

impl std::error::Error for MixPointError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_valid_mix_point() {
        let mp = MixPoint::new(44100, 4, 0.85, MixType::Outro).unwrap();
        assert_eq!(mp.sample_position(), 44100);
        assert_eq!(mp.beat_index(), 4);
        assert!((mp.confidence() - 0.85).abs() < 1e-10);
        assert_eq!(mp.mix_type(), MixType::Outro);
    }

    #[test]
    fn confidence_at_boundaries() {
        assert!(MixPoint::new(0, 0, 0.0, MixType::Intro).is_ok());
        assert!(MixPoint::new(0, 0, 1.0, MixType::Intro).is_ok());
    }

    #[test]
    fn reject_confidence_above_one() {
        assert!(MixPoint::new(0, 0, 1.1, MixType::Intro).is_err());
    }

    #[test]
    fn reject_negative_confidence() {
        assert!(MixPoint::new(0, 0, -0.1, MixType::Intro).is_err());
    }

    #[test]
    fn reject_nan_confidence() {
        assert!(MixPoint::new(0, 0, f64::NAN, MixType::Intro).is_err());
    }

    #[test]
    fn all_mix_types() {
        assert_eq!(
            MixPoint::new(0, 0, 0.5, MixType::Intro).unwrap().mix_type(),
            MixType::Intro
        );
        assert_eq!(
            MixPoint::new(0, 0, 0.5, MixType::Outro).unwrap().mix_type(),
            MixType::Outro
        );
        assert_eq!(
            MixPoint::new(0, 0, 0.5, MixType::Breakdown).unwrap().mix_type(),
            MixType::Breakdown
        );
        assert_eq!(
            MixPoint::new(0, 0, 0.5, MixType::Drop).unwrap().mix_type(),
            MixType::Drop
        );
    }
}
