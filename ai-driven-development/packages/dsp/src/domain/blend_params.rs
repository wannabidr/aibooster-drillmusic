//! Parameters and types for AI-powered audio blending.

/// 3-band EQ frequency band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EqBand {
    Low,
    Mid,
    High,
}

/// EQ automation curve: how each band's gain changes over the transition.
#[derive(Debug, Clone)]
pub struct EqAutomation {
    /// Gain curves for track A: (low, mid, high) at each normalized position [0..1].
    /// Each value is a gain multiplier (0.0 = silent, 1.0 = unity).
    pub track_a_curves: BandCurves,
    /// Gain curves for track B.
    pub track_b_curves: BandCurves,
}

/// Gain curves for the three EQ bands.
#[derive(Debug, Clone)]
pub struct BandCurves {
    pub low: AutomationCurve,
    pub mid: AutomationCurve,
    pub high: AutomationCurve,
}

/// An automation curve defined by a series of control points.
/// Interpolation is linear between points.
#[derive(Debug, Clone)]
pub struct AutomationCurve {
    /// (position, value) pairs where position is in [0.0, 1.0] and value is gain.
    points: Vec<(f64, f64)>,
}

impl AutomationCurve {
    pub fn new(points: Vec<(f64, f64)>) -> Result<Self, BlendParamsError> {
        if points.len() < 2 {
            return Err(BlendParamsError::InsufficientPoints);
        }
        // Verify sorted by position
        for w in points.windows(2) {
            if w[0].0 >= w[1].0 {
                return Err(BlendParamsError::UnsortedPoints);
            }
        }
        Ok(Self { points })
    }

    /// Create a simple ramp from `start_val` to `end_val`.
    pub fn ramp(start_val: f64, end_val: f64) -> Self {
        Self {
            points: vec![(0.0, start_val), (1.0, end_val)],
        }
    }

    /// Create a constant value curve.
    pub fn constant(value: f64) -> Self {
        Self {
            points: vec![(0.0, value), (1.0, value)],
        }
    }

    /// Evaluate the curve at a normalized position [0.0, 1.0].
    pub fn value_at(&self, position: f64) -> f64 {
        let position = position.clamp(0.0, 1.0);

        if position <= self.points[0].0 {
            return self.points[0].1;
        }
        if position >= self.points[self.points.len() - 1].0 {
            return self.points[self.points.len() - 1].1;
        }

        // Find the two surrounding points and interpolate
        for w in self.points.windows(2) {
            if position >= w[0].0 && position <= w[1].0 {
                let t = (position - w[0].0) / (w[1].0 - w[0].0);
                return w[0].1 + t * (w[1].1 - w[0].1);
            }
        }

        self.points.last().unwrap().1
    }

    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }
}

/// Filter type for sweep automation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    HighPass,
    LowPass,
}

/// Parameters for a filter sweep.
#[derive(Debug, Clone)]
pub struct FilterSweepParams {
    pub filter_type: FilterType,
    /// Automation curve for cutoff frequency (Hz), normalized position [0..1].
    pub cutoff_curve: AutomationCurve,
    /// Resonance (Q factor), typically 0.5 to 10.0.
    pub resonance: f64,
}

impl FilterSweepParams {
    pub fn new(filter_type: FilterType, cutoff_curve: AutomationCurve, resonance: f64) -> Result<Self, BlendParamsError> {
        if resonance <= 0.0 || !resonance.is_finite() {
            return Err(BlendParamsError::InvalidResonance(resonance));
        }
        Ok(Self { filter_type, cutoff_curve, resonance })
    }
}

/// A detected phrase within a track.
#[derive(Debug, Clone)]
pub struct Phrase {
    /// Start beat index in the beat grid.
    pub start_beat: usize,
    /// End beat index (exclusive).
    pub end_beat: usize,
    /// Average energy level of this phrase (0.0 to 1.0).
    pub energy: f64,
    /// Type of phrase segment.
    pub phrase_type: PhraseType,
}

impl Phrase {
    pub fn length_beats(&self) -> usize {
        self.end_beat.saturating_sub(self.start_beat)
    }
}

/// Classification of a phrase segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhraseType {
    Intro,
    Buildup,
    Drop,
    Breakdown,
    Outro,
    Body,
}

/// User-selectable blend style presets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendStyle {
    /// Long gradual blend with EQ swapping (32 beats).
    LongBlend,
    /// Short cut transition (4-8 beats).
    ShortCut,
    /// Echo/delay tail on outgoing track.
    EchoOut,
    /// Filter sweep on outgoing track.
    FilterSweep,
    /// Backspin effect on outgoing track.
    Backspin,
}

impl BlendStyle {
    /// Default transition length in beats for each style.
    pub fn default_beats(&self) -> u32 {
        match self {
            BlendStyle::LongBlend => 32,
            BlendStyle::ShortCut => 4,
            BlendStyle::EchoOut => 16,
            BlendStyle::FilterSweep => 16,
            BlendStyle::Backspin => 8,
        }
    }

    pub fn parse(s: &str) -> Result<Self, BlendParamsError> {
        match s.to_lowercase().as_str() {
            "long_blend" | "long-blend" | "longblend" => Ok(BlendStyle::LongBlend),
            "short_cut" | "short-cut" | "shortcut" => Ok(BlendStyle::ShortCut),
            "echo_out" | "echo-out" | "echoout" => Ok(BlendStyle::EchoOut),
            "filter_sweep" | "filter-sweep" | "filtersweep" => Ok(BlendStyle::FilterSweep),
            "backspin" => Ok(BlendStyle::Backspin),
            other => Err(BlendParamsError::InvalidBlendStyle(other.to_string())),
        }
    }
}

/// Genre hint for blend style selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Genre {
    House,
    Techno,
    DrumAndBass,
    Trance,
    HipHop,
    Pop,
    Unknown,
}

impl Genre {
    /// Suggest the best blend style for this genre.
    pub fn suggested_blend_style(&self) -> BlendStyle {
        match self {
            Genre::House | Genre::Techno => BlendStyle::LongBlend,
            Genre::Trance => BlendStyle::FilterSweep,
            Genre::DrumAndBass => BlendStyle::ShortCut,
            Genre::HipHop => BlendStyle::EchoOut,
            Genre::Pop => BlendStyle::FilterSweep,
            Genre::Unknown => BlendStyle::LongBlend,
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "house" => Genre::House,
            "techno" => Genre::Techno,
            "drum_and_bass" | "dnb" | "drum-and-bass" => Genre::DrumAndBass,
            "trance" => Genre::Trance,
            "hip_hop" | "hiphop" | "hip-hop" => Genre::HipHop,
            "pop" => Genre::Pop,
            _ => Genre::Unknown,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlendParamsError {
    InsufficientPoints,
    UnsortedPoints,
    InvalidResonance(f64),
    InvalidBlendStyle(String),
}

impl std::fmt::Display for BlendParamsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientPoints => write!(f, "automation curve needs at least 2 points"),
            Self::UnsortedPoints => write!(f, "automation curve points must be sorted by position"),
            Self::InvalidResonance(r) => write!(f, "invalid resonance: {r}"),
            Self::InvalidBlendStyle(s) => write!(f, "invalid blend style: '{s}'"),
        }
    }
}

impl std::error::Error for BlendParamsError {}

#[cfg(test)]
mod tests {
    use super::*;

    // -- AutomationCurve tests --

    #[test]
    fn ramp_curve_start_and_end() {
        let curve = AutomationCurve::ramp(1.0, 0.0);
        assert!((curve.value_at(0.0) - 1.0).abs() < 1e-10);
        assert!((curve.value_at(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn ramp_curve_midpoint() {
        let curve = AutomationCurve::ramp(0.0, 1.0);
        assert!((curve.value_at(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn constant_curve_is_flat() {
        let curve = AutomationCurve::constant(0.7);
        for i in 0..=10 {
            let pos = i as f64 / 10.0;
            assert!((curve.value_at(pos) - 0.7).abs() < 1e-10);
        }
    }

    #[test]
    fn multi_point_curve() {
        let curve = AutomationCurve::new(vec![
            (0.0, 1.0),
            (0.5, 0.0),
            (1.0, 1.0),
        ]).unwrap();
        assert!((curve.value_at(0.0) - 1.0).abs() < 1e-10);
        assert!((curve.value_at(0.25) - 0.5).abs() < 1e-10);
        assert!((curve.value_at(0.5) - 0.0).abs() < 1e-10);
        assert!((curve.value_at(0.75) - 0.5).abs() < 1e-10);
        assert!((curve.value_at(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn curve_clamps_below_zero() {
        let curve = AutomationCurve::ramp(1.0, 0.0);
        assert!((curve.value_at(-0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn curve_clamps_above_one() {
        let curve = AutomationCurve::ramp(1.0, 0.0);
        assert!((curve.value_at(1.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn reject_insufficient_points() {
        assert!(AutomationCurve::new(vec![(0.5, 1.0)]).is_err());
    }

    #[test]
    fn reject_unsorted_points() {
        assert!(AutomationCurve::new(vec![(0.5, 1.0), (0.2, 0.0)]).is_err());
    }

    // -- BlendStyle tests --

    #[test]
    fn blend_style_default_beats() {
        assert_eq!(BlendStyle::LongBlend.default_beats(), 32);
        assert_eq!(BlendStyle::ShortCut.default_beats(), 4);
        assert_eq!(BlendStyle::EchoOut.default_beats(), 16);
        assert_eq!(BlendStyle::FilterSweep.default_beats(), 16);
        assert_eq!(BlendStyle::Backspin.default_beats(), 8);
    }

    #[test]
    fn blend_style_from_str() {
        assert_eq!(BlendStyle::parse("long_blend").unwrap(), BlendStyle::LongBlend);
        assert_eq!(BlendStyle::parse("short-cut").unwrap(), BlendStyle::ShortCut);
        assert_eq!(BlendStyle::parse("echoout").unwrap(), BlendStyle::EchoOut);
        assert_eq!(BlendStyle::parse("filter_sweep").unwrap(), BlendStyle::FilterSweep);
        assert_eq!(BlendStyle::parse("backspin").unwrap(), BlendStyle::Backspin);
        assert!(BlendStyle::parse("invalid").is_err());
    }

    // -- Genre tests --

    #[test]
    fn genre_suggested_styles() {
        assert_eq!(Genre::House.suggested_blend_style(), BlendStyle::LongBlend);
        assert_eq!(Genre::Techno.suggested_blend_style(), BlendStyle::LongBlend);
        assert_eq!(Genre::Trance.suggested_blend_style(), BlendStyle::FilterSweep);
        assert_eq!(Genre::DrumAndBass.suggested_blend_style(), BlendStyle::ShortCut);
        assert_eq!(Genre::HipHop.suggested_blend_style(), BlendStyle::EchoOut);
    }

    #[test]
    fn genre_from_str() {
        assert_eq!(Genre::parse("house"), Genre::House);
        assert_eq!(Genre::parse("techno"), Genre::Techno);
        assert_eq!(Genre::parse("dnb"), Genre::DrumAndBass);
        assert_eq!(Genre::parse("unknown_genre"), Genre::Unknown);
    }

    // -- FilterSweepParams tests --

    #[test]
    fn valid_filter_params() {
        let curve = AutomationCurve::ramp(200.0, 20000.0);
        let params = FilterSweepParams::new(FilterType::HighPass, curve, 1.0).unwrap();
        assert_eq!(params.filter_type, FilterType::HighPass);
        assert!((params.resonance - 1.0).abs() < 1e-10);
    }

    #[test]
    fn reject_zero_resonance() {
        let curve = AutomationCurve::ramp(200.0, 20000.0);
        assert!(FilterSweepParams::new(FilterType::LowPass, curve, 0.0).is_err());
    }

    #[test]
    fn reject_negative_resonance() {
        let curve = AutomationCurve::ramp(200.0, 20000.0);
        assert!(FilterSweepParams::new(FilterType::LowPass, curve, -1.0).is_err());
    }

    // -- Phrase tests --

    #[test]
    fn phrase_length_beats() {
        let phrase = Phrase {
            start_beat: 0,
            end_beat: 32,
            energy: 0.3,
            phrase_type: PhraseType::Intro,
        };
        assert_eq!(phrase.length_beats(), 32);
    }
}
