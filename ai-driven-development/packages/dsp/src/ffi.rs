//! FFI exports for Tauri integration.
//!
//! These functions accept simple types (strings, integers) suitable for
//! calling from Tauri `#[tauri::command]` wrappers. File I/O and format
//! conversion happens here, keeping the engine layer pure.

use crate::blend::{AiBlendConfig, AiBlendEngine};
use crate::domain::{BeatGrid, BlendStyle, CrossfadeCurve, CrossfadeStyle, Genre, TransitionParams};
use crate::engine::CrossfadeEngine;
use crate::io;
use std::path::PathBuf;

/// Errors from FFI operations.
#[derive(Debug)]
pub enum FfiError {
    Io(io::IoError),
    Engine(crate::engine::CrossfadeError),
    AiBlend(crate::blend::AiBlendError),
    InvalidCurve(String),
    InvalidPath(String),
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Engine(e) => write!(f, "{e}"),
            Self::AiBlend(e) => write!(f, "{e}"),
            Self::InvalidCurve(c) => write!(f, "invalid curve type: '{c}' (expected 'linear' or 'equal_power')"),
            Self::InvalidPath(p) => write!(f, "invalid path: {p}"),
        }
    }
}

impl std::error::Error for FfiError {}

impl From<io::IoError> for FfiError {
    fn from(e: io::IoError) -> Self {
        FfiError::Io(e)
    }
}

impl From<crate::engine::CrossfadeError> for FfiError {
    fn from(e: crate::engine::CrossfadeError) -> Self {
        FfiError::Engine(e)
    }
}

impl From<crate::blend::AiBlendError> for FfiError {
    fn from(e: crate::blend::AiBlendError) -> Self {
        FfiError::AiBlend(e)
    }
}

/// Parse a curve type string into a CrossfadeCurve.
fn parse_curve(curve: &str) -> Result<CrossfadeCurve, FfiError> {
    match curve.to_lowercase().as_str() {
        "linear" => Ok(CrossfadeCurve::Linear),
        "equal_power" | "equalpower" | "equal-power" => Ok(CrossfadeCurve::EqualPower),
        other => Err(FfiError::InvalidCurve(other.to_string())),
    }
}

/// Generate an output path for the rendered preview.
fn output_path(track_a_path: &str, track_b_path: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ai_dj_previews");
    std::fs::create_dir_all(&dir).ok();

    // Create a deterministic filename from input paths
    let hash = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        track_a_path.hash(&mut hasher);
        track_b_path.hash(&mut hasher);
        hasher.finish()
    };
    dir.join(format!("preview_{hash:016x}.wav"))
}

/// Render a crossfade preview between two WAV files.
///
/// This is the main FFI entry point, designed to be wrapped by `#[tauri::command]`.
///
/// # Arguments
/// * `track_a_path` - Path to the outgoing track (WAV)
/// * `track_b_path` - Path to the incoming track (WAV)
/// * `bpm_a` - BPM of track A (from Python analysis)
/// * `bpm_b` - BPM of track B (from Python analysis)
/// * `transition_beats` - Number of beats for the crossfade (16 or 32)
/// * `curve` - Curve type: "linear" or "equal_power"
///
/// # Returns
/// Path to the rendered preview WAV file.
pub fn render_crossfade(
    track_a_path: &str,
    track_b_path: &str,
    bpm_a: f64,
    bpm_b: f64,
    transition_beats: u32,
    curve: &str,
) -> Result<String, FfiError> {
    // Parse curve type
    let crossfade_curve = parse_curve(curve)?;

    // Load audio files
    let track_a = io::load_wav(track_a_path)?;
    let track_b = io::load_wav(track_b_path)?;

    // Generate beat grids from BPM data
    let grid_a = BeatGrid::from_bpm(bpm_a, 0, track_a.sample_rate(), track_a.num_frames() as u64)
        .map_err(|_| FfiError::Engine(crate::engine::CrossfadeError::BeatGridMissing { track: "A" }))?;
    let grid_b = BeatGrid::from_bpm(bpm_b, 0, track_b.sample_rate(), track_b.num_frames() as u64)
        .map_err(|_| FfiError::Engine(crate::engine::CrossfadeError::BeatGridMissing { track: "B" }))?;

    // Select mix points: default to 75% through track A, start of track B
    let mix_point_a = {
        let preferred = (track_a.num_frames() as f64 * 0.75) as u64;
        CrossfadeEngine::find_mix_point(&grid_a, preferred).unwrap_or(preferred)
    };
    let mix_point_b = 0u64;

    // Build transition params
    let style = CrossfadeStyle::VolumeFade(crossfade_curve);
    let params = TransitionParams::new(transition_beats, style, mix_point_a, mix_point_b)
        .map_err(|_| FfiError::InvalidCurve("invalid transition beats".to_string()))?;

    // Render crossfade with 8 beats of context on each side
    let result = CrossfadeEngine::render(
        &track_a, &track_b, &grid_a, &grid_b, &params, 8,
    )?;

    // Save output
    let out_path = output_path(track_a_path, track_b_path);
    io::save_wav(result.buffer(), &out_path)?;

    Ok(out_path.to_string_lossy().to_string())
}

/// Parameters for rendering a crossfade with explicit mix points.
pub struct RenderAtParams<'a> {
    pub track_a_path: &'a str,
    pub track_b_path: &'a str,
    pub bpm_a: f64,
    pub bpm_b: f64,
    pub mix_point_a_secs: f64,
    pub mix_point_b_secs: f64,
    pub transition_beats: u32,
    pub curve: &'a str,
}

/// Render a crossfade preview with explicit mix points.
///
/// Like `render_crossfade` but allows specifying exact mix point positions
/// (in seconds) instead of using defaults.
pub fn render_crossfade_at(p: &RenderAtParams<'_>) -> Result<String, FfiError> {
    let crossfade_curve = parse_curve(p.curve)?;

    let track_a = io::load_wav(p.track_a_path)?;
    let track_b = io::load_wav(p.track_b_path)?;

    let grid_a = BeatGrid::from_bpm(p.bpm_a, 0, track_a.sample_rate(), track_a.num_frames() as u64)
        .map_err(|_| FfiError::Engine(crate::engine::CrossfadeError::BeatGridMissing { track: "A" }))?;
    let grid_b = BeatGrid::from_bpm(p.bpm_b, 0, track_b.sample_rate(), track_b.num_frames() as u64)
        .map_err(|_| FfiError::Engine(crate::engine::CrossfadeError::BeatGridMissing { track: "B" }))?;

    // Convert seconds to sample positions and snap to downbeats
    let raw_a = (p.mix_point_a_secs * track_a.sample_rate() as f64) as u64;
    let raw_b = (p.mix_point_b_secs * track_b.sample_rate() as f64) as u64;
    let mix_point_a = CrossfadeEngine::find_mix_point(&grid_a, raw_a).unwrap_or(raw_a);
    let mix_point_b = CrossfadeEngine::find_mix_point(&grid_b, raw_b).unwrap_or(raw_b);

    let style = CrossfadeStyle::VolumeFade(crossfade_curve);
    let params = TransitionParams::new(p.transition_beats, style, mix_point_a, mix_point_b)
        .map_err(|_| FfiError::InvalidCurve("invalid transition beats".to_string()))?;

    let result = CrossfadeEngine::render(
        &track_a, &track_b, &grid_a, &grid_b, &params, 8,
    )?;

    let out_path = output_path(p.track_a_path, p.track_b_path);
    io::save_wav(result.buffer(), &out_path)?;

    Ok(out_path.to_string_lossy().to_string())
}

/// Render an AI-powered blend preview between two WAV files.
///
/// This is the Phase 2 FFI entry point for Tauri integration.
///
/// # Arguments
/// * `track_a_path` - Path to the outgoing track (WAV)
/// * `track_b_path` - Path to the incoming track (WAV)
/// * `bpm_a` - BPM of track A
/// * `bpm_b` - BPM of track B
/// * `blend_style` - Style: "long_blend", "short_cut", "echo_out", "filter_sweep", "backspin"
/// * `genre` - Genre hint: "house", "techno", "trance", "dnb", "hip_hop", "pop"
///
/// # Returns
/// Path to the rendered preview WAV file.
pub fn render_ai_blend(
    track_a_path: &str,
    track_b_path: &str,
    bpm_a: f64,
    bpm_b: f64,
    blend_style: &str,
    genre: &str,
) -> Result<String, FfiError> {
    let style = BlendStyle::parse(blend_style)
        .map_err(|_| FfiError::InvalidCurve(format!("invalid blend style: '{blend_style}'")))?;
    let genre_hint = Genre::parse(genre);

    let track_a = io::load_wav(track_a_path)?;
    let track_b = io::load_wav(track_b_path)?;

    let grid_a = BeatGrid::from_bpm(bpm_a, 0, track_a.sample_rate(), track_a.num_frames() as u64)
        .map_err(|_| FfiError::Engine(crate::engine::CrossfadeError::BeatGridMissing { track: "A" }))?;
    let grid_b = BeatGrid::from_bpm(bpm_b, 0, track_b.sample_rate(), track_b.num_frames() as u64)
        .map_err(|_| FfiError::Engine(crate::engine::CrossfadeError::BeatGridMissing { track: "B" }))?;

    let config = AiBlendConfig::new(style, genre_hint);
    let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 8)?;

    let out_path = output_path(track_a_path, track_b_path);
    io::save_wav(result.buffer(), &out_path)?;

    Ok(out_path.to_string_lossy().to_string())
}

/// Render an AI-powered blend with auto-detected style based on genre.
pub fn render_ai_blend_auto(
    track_a_path: &str,
    track_b_path: &str,
    bpm_a: f64,
    bpm_b: f64,
    genre: &str,
) -> Result<String, FfiError> {
    let genre_hint = Genre::parse(genre);
    let style = AiBlendEngine::auto_select_style(genre_hint, bpm_a, bpm_b);
    let style_str = match style {
        BlendStyle::LongBlend => "long_blend",
        BlendStyle::ShortCut => "short_cut",
        BlendStyle::EchoOut => "echo_out",
        BlendStyle::FilterSweep => "filter_sweep",
        BlendStyle::Backspin => "backspin",
    };
    render_ai_blend(track_a_path, track_b_path, bpm_a, bpm_b, style_str, genre)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::AudioBuffer;
    use std::fs;
    use std::path::Path;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join("ai_dj_dsp_ffi_tests");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Create a test WAV file with a sine wave.
    fn create_test_wav(path: &Path, duration_secs: f32, sample_rate: u32) {
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        let buf = AudioBuffer::new(samples, sample_rate, 1).unwrap();
        io::save_wav(&buf, path).unwrap();
    }

    #[test]
    fn parse_curve_linear() {
        assert_eq!(parse_curve("linear").unwrap(), CrossfadeCurve::Linear);
    }

    #[test]
    fn parse_curve_equal_power_variants() {
        assert_eq!(parse_curve("equal_power").unwrap(), CrossfadeCurve::EqualPower);
        assert_eq!(parse_curve("equalpower").unwrap(), CrossfadeCurve::EqualPower);
        assert_eq!(parse_curve("equal-power").unwrap(), CrossfadeCurve::EqualPower);
        assert_eq!(parse_curve("EQUAL_POWER").unwrap(), CrossfadeCurve::EqualPower);
    }

    #[test]
    fn parse_curve_invalid() {
        assert!(parse_curve("invalid").is_err());
    }

    #[test]
    fn render_crossfade_full_pipeline() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_track_a.wav");
        let track_b = dir.join("ffi_track_b.wav");

        create_test_wav(&track_a, 30.0, 44100);
        create_test_wav(&track_b, 30.0, 44100);

        let result = render_crossfade(
            track_a.to_str().unwrap(),
            track_b.to_str().unwrap(),
            128.0,
            128.0,
            16,
            "linear",
        );

        assert!(result.is_ok(), "render failed: {:?}", result.err());
        let output_path = result.unwrap();
        assert!(Path::new(&output_path).exists(), "output file should exist");

        // Verify output is a valid WAV
        let loaded = io::load_wav(&output_path).unwrap();
        assert_eq!(loaded.sample_rate(), 44100);
        assert!(loaded.num_frames() > 0);

        // Cleanup
        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn render_crossfade_equal_power_pipeline() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_ep_track_a.wav");
        let track_b = dir.join("ffi_ep_track_b.wav");

        // 120s tracks needed: 75% mix point at 90s + 32-beat transition (~15s) + context
        create_test_wav(&track_a, 120.0, 44100);
        create_test_wav(&track_b, 120.0, 44100);

        let result = render_crossfade(
            track_a.to_str().unwrap(),
            track_b.to_str().unwrap(),
            126.0,
            128.0,
            32,
            "equal_power",
        );

        assert!(result.is_ok(), "render failed: {:?}", result.err());

        let output_path = result.unwrap();
        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn render_crossfade_at_explicit_points() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_at_track_a.wav");
        let track_b = dir.join("ffi_at_track_b.wav");

        create_test_wav(&track_a, 60.0, 44100);
        create_test_wav(&track_b, 60.0, 44100);

        let result = render_crossfade_at(&RenderAtParams {
            track_a_path: track_a.to_str().unwrap(),
            track_b_path: track_b.to_str().unwrap(),
            bpm_a: 128.0,
            bpm_b: 128.0,
            mix_point_a_secs: 40.0,
            mix_point_b_secs: 5.0,
            transition_beats: 16,
            curve: "linear",
        });

        assert!(result.is_ok(), "render failed: {:?}", result.err());

        let output_path = result.unwrap();
        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn render_crossfade_nonexistent_file() {
        let result = render_crossfade(
            "/nonexistent/track_a.wav",
            "/nonexistent/track_b.wav",
            128.0,
            128.0,
            16,
            "linear",
        );
        assert!(result.is_err());
    }

    #[test]
    fn render_crossfade_invalid_curve() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_inv_track_a.wav");
        let track_b = dir.join("ffi_inv_track_b.wav");

        create_test_wav(&track_a, 10.0, 44100);
        create_test_wav(&track_b, 10.0, 44100);

        let result = render_crossfade(
            track_a.to_str().unwrap(),
            track_b.to_str().unwrap(),
            128.0,
            128.0,
            16,
            "invalid_curve",
        );
        assert!(result.is_err());

        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
    }

    #[test]
    fn render_ai_blend_long_blend() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_ai_track_a.wav");
        let track_b = dir.join("ffi_ai_track_b.wav");

        create_test_wav(&track_a, 60.0, 44100);
        create_test_wav(&track_b, 60.0, 44100);

        let result = render_ai_blend(
            track_a.to_str().unwrap(),
            track_b.to_str().unwrap(),
            128.0,
            128.0,
            "long_blend",
            "house",
        );

        assert!(result.is_ok(), "AI blend failed: {:?}", result.err());
        let output_path = result.unwrap();
        assert!(Path::new(&output_path).exists());

        let loaded = io::load_wav(&output_path).unwrap();
        assert!(loaded.num_frames() > 0);

        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn render_ai_blend_auto_selects_style() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_auto_track_a.wav");
        let track_b = dir.join("ffi_auto_track_b.wav");

        create_test_wav(&track_a, 60.0, 44100);
        create_test_wav(&track_b, 60.0, 44100);

        let result = render_ai_blend_auto(
            track_a.to_str().unwrap(),
            track_b.to_str().unwrap(),
            128.0,
            128.0,
            "techno",
        );

        assert!(result.is_ok(), "AI blend auto failed: {:?}", result.err());

        let output_path = result.unwrap();
        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
        fs::remove_file(&output_path).ok();
    }

    #[test]
    fn render_ai_blend_invalid_style() {
        let dir = temp_dir();
        let track_a = dir.join("ffi_inv_style_a.wav");
        let track_b = dir.join("ffi_inv_style_b.wav");

        create_test_wav(&track_a, 10.0, 44100);
        create_test_wav(&track_b, 10.0, 44100);

        let result = render_ai_blend(
            track_a.to_str().unwrap(),
            track_b.to_str().unwrap(),
            128.0,
            128.0,
            "invalid_style",
            "house",
        );
        assert!(result.is_err());

        fs::remove_file(&track_a).ok();
        fs::remove_file(&track_b).ok();
    }
}
