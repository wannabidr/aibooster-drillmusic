use crate::domain::{
    AudioBuffer, AudioBufferError, BeatGrid,
    RenderMetadata, RenderResult, TransitionParams,
};
use std::time::Instant;

/// Errors that can occur during crossfade rendering.
#[derive(Debug, Clone, PartialEq)]
pub enum CrossfadeError {
    /// Track audio is too short for the requested transition.
    TrackTooShort { track: &'static str, available_frames: usize, required_frames: usize },
    /// Sample rates don't match between tracks.
    SampleRateMismatch { rate_a: u32, rate_b: u32 },
    /// Channel count doesn't match between tracks.
    ChannelMismatch { ch_a: u16, ch_b: u16 },
    /// Beat grid error during rendering.
    BeatGridMissing { track: &'static str },
    /// Audio buffer construction failed.
    BufferError(AudioBufferError),
}

impl std::fmt::Display for CrossfadeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TrackTooShort { track, available_frames, required_frames } => {
                write!(f, "track {track} too short: {available_frames} frames available, {required_frames} required")
            }
            Self::SampleRateMismatch { rate_a, rate_b } => {
                write!(f, "sample rate mismatch: track A={rate_a}Hz, track B={rate_b}Hz")
            }
            Self::ChannelMismatch { ch_a, ch_b } => {
                write!(f, "channel mismatch: track A={ch_a}, track B={ch_b}")
            }
            Self::BeatGridMissing { track } => {
                write!(f, "beat grid missing for track {track}")
            }
            Self::BufferError(e) => write!(f, "buffer error: {e}"),
        }
    }
}

impl std::error::Error for CrossfadeError {}

impl From<AudioBufferError> for CrossfadeError {
    fn from(e: AudioBufferError) -> Self {
        CrossfadeError::BufferError(e)
    }
}

/// The crossfade engine renders beat-aligned audio transitions.
pub struct CrossfadeEngine;

impl CrossfadeEngine {
    /// Render a crossfade between two audio buffers using the given parameters.
    ///
    /// The output contains:
    /// - Pre-transition audio from track A (context before the fade)
    /// - The crossfade transition region
    /// - Post-transition audio from track B (context after the fade)
    ///
    /// `context_beats` controls how many beats of solo audio to include
    /// before and after the transition.
    pub fn render(
        track_a: &AudioBuffer,
        track_b: &AudioBuffer,
        grid_a: &BeatGrid,
        grid_b: &BeatGrid,
        params: &TransitionParams,
        context_beats: u32,
    ) -> Result<RenderResult, CrossfadeError> {
        let start_time = Instant::now();

        // Validate matching formats
        if track_a.sample_rate() != track_b.sample_rate() {
            return Err(CrossfadeError::SampleRateMismatch {
                rate_a: track_a.sample_rate(),
                rate_b: track_b.sample_rate(),
            });
        }
        if track_a.channels() != track_b.channels() {
            return Err(CrossfadeError::ChannelMismatch {
                ch_a: track_a.channels(),
                ch_b: track_b.channels(),
            });
        }

        let sample_rate = track_a.sample_rate();
        let channels = track_a.channels() as usize;
        let samples_per_beat_a = (sample_rate as f64 * 60.0 / grid_a.bpm()) as usize;
        let samples_per_beat_b = (sample_rate as f64 * 60.0 / grid_b.bpm()) as usize;
        let transition_frames = samples_per_beat_a * params.transition_beats() as usize;

        // Determine mix points (in frames)
        let mix_point_a = params.mix_point_a() as usize;
        let mix_point_b = params.mix_point_b() as usize;

        // Context: beats of solo audio before/after transition
        let context_frames_a = samples_per_beat_a * context_beats as usize;
        let context_frames_b = samples_per_beat_b * context_beats as usize;

        // Calculate required regions
        let a_start = mix_point_a.saturating_sub(context_frames_a);
        let a_end = mix_point_a + transition_frames;
        let b_end = mix_point_b + transition_frames + context_frames_b;

        if a_end > track_a.num_frames() {
            return Err(CrossfadeError::TrackTooShort {
                track: "A",
                available_frames: track_a.num_frames(),
                required_frames: a_end,
            });
        }
        if b_end > track_b.num_frames() {
            return Err(CrossfadeError::TrackTooShort {
                track: "B",
                available_frames: track_b.num_frames(),
                required_frames: b_end,
            });
        }

        let total_frames = context_frames_a + transition_frames + context_frames_b;
        let mut output = vec![0.0f32; total_frames * channels];

        let curve = params.curve_type();

        // Region 1: Pre-transition context (solo track A)
        let samples_a = track_a.samples();
        for frame in 0..context_frames_a {
            let src_frame = a_start + frame;
            for ch in 0..channels {
                output[frame * channels + ch] = samples_a[src_frame * channels + ch];
            }
        }

        // Region 2: Crossfade transition
        let samples_b = track_b.samples();
        for frame in 0..transition_frames {
            let position = if transition_frames <= 1 {
                1.0
            } else {
                frame as f64 / (transition_frames - 1) as f64
            };
            let (gain_a, gain_b) = curve.gain_at(position);

            let src_a_frame = mix_point_a + frame;
            let src_b_frame = mix_point_b + frame;
            let out_frame = context_frames_a + frame;

            for ch in 0..channels {
                let sample_a = samples_a[src_a_frame * channels + ch] * gain_a as f32;
                let sample_b = samples_b[src_b_frame * channels + ch] * gain_b as f32;
                output[out_frame * channels + ch] = sample_a + sample_b;
            }
        }

        // Region 3: Post-transition context (solo track B)
        for frame in 0..context_frames_b {
            let src_frame = mix_point_b + transition_frames + frame;
            let out_frame = context_frames_a + transition_frames + frame;
            for ch in 0..channels {
                output[out_frame * channels + ch] = samples_b[src_frame * channels + ch];
            }
        }

        let buffer = AudioBuffer::new(output, sample_rate, track_a.channels())?;
        let render_time_ms = start_time.elapsed().as_millis() as u64;

        let transition_start_sample = context_frames_a as u64;
        let transition_end_sample = (context_frames_a + transition_frames) as u64;

        let metadata = RenderMetadata {
            render_time_ms,
            track_a_bpm: grid_a.bpm(),
            track_b_bpm: grid_b.bpm(),
            curve_type: curve,
            transition_beats: params.transition_beats(),
        };

        Ok(RenderResult::new(
            buffer,
            transition_start_sample,
            transition_end_sample,
            metadata,
        ))
    }

    /// Render a simple crossfade without context beats (transition region only).
    pub fn render_transition_only(
        track_a: &AudioBuffer,
        track_b: &AudioBuffer,
        grid_a: &BeatGrid,
        grid_b: &BeatGrid,
        params: &TransitionParams,
    ) -> Result<RenderResult, CrossfadeError> {
        Self::render(track_a, track_b, grid_a, grid_b, params, 0)
    }

    /// Find the best mix-out point in track A (nearest downbeat to the requested position).
    pub fn find_mix_point(
        grid: &BeatGrid,
        preferred_sample: u64,
    ) -> Option<u64> {
        let downbeat_idx = grid.nearest_downbeat(preferred_sample)?;
        Some(grid.beat_positions()[downbeat_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{CrossfadeCurve, CrossfadeStyle, TransitionParams};

    /// Helper: generate a constant-value mono buffer.
    fn constant_buffer(value: f32, num_frames: usize, sample_rate: u32) -> AudioBuffer {
        AudioBuffer::new(vec![value; num_frames], sample_rate, 1).unwrap()
    }

    /// Helper: create a simple beat grid for a buffer.
    fn simple_grid(bpm: f64, sample_rate: u32, total_frames: u64) -> BeatGrid {
        BeatGrid::from_bpm(bpm, 0, sample_rate, total_frames).unwrap()
    }

    #[test]
    fn render_linear_crossfade_transition_only() {
        let sr = 44100;
        let bpm = 120.0;
        let track_a = constant_buffer(1.0, sr * 10, sr as u32);
        let track_b = constant_buffer(0.5, sr * 10, sr as u32);
        let grid_a = simple_grid(bpm, sr as u32, sr as u64 * 10);
        let grid_b = simple_grid(bpm, sr as u32, sr as u64 * 10);

        // 4-beat transition starting at frame 0
        let params = TransitionParams::new(
            4,
            CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear),
            0,
            0,
        ).unwrap();

        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        let buf = result.buffer();
        assert!(buf.num_frames() > 0);
        assert_eq!(buf.sample_rate(), sr as u32);
        assert_eq!(result.transition_start_sample(), 0);

        // First sample: 100% track A (1.0), 0% track B
        assert!((buf.samples()[0] - 1.0).abs() < 0.01);
        // Last sample: 0% track A, 100% track B (0.5)
        let last = buf.samples()[buf.num_frames() - 1];
        assert!((last - 0.5).abs() < 0.01);
    }

    #[test]
    fn render_equal_power_crossfade() {
        let sr = 44100u32;
        let bpm = 128.0;
        let track_a = constant_buffer(1.0, sr as usize * 10, sr);
        let track_b = constant_buffer(1.0, sr as usize * 10, sr);
        let grid_a = simple_grid(bpm, sr, sr as u64 * 10);
        let grid_b = simple_grid(bpm, sr, sr as u64 * 10);

        let params = TransitionParams::new(
            4,
            CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower),
            0,
            0,
        ).unwrap();

        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        // With equal-power and both tracks at 1.0, midpoint should be ~1.414 (sqrt(2))
        let mid = result.buffer().num_frames() / 2;
        let mid_val = result.buffer().samples()[mid];
        let expected = 2.0_f32.sqrt(); // cos(pi/4) + sin(pi/4) = 2/sqrt(2) = sqrt(2)
        assert!(
            (mid_val - expected).abs() < 0.05,
            "midpoint value {mid_val} should be near {expected}"
        );
    }

    #[test]
    fn render_with_context_beats() {
        let sr = 44100u32;
        let bpm = 120.0;
        let frames = sr as usize * 30;
        let track_a = constant_buffer(1.0, frames, sr);
        let track_b = constant_buffer(0.5, frames, sr);
        let grid_a = simple_grid(bpm, sr, frames as u64);
        let grid_b = simple_grid(bpm, sr, frames as u64);

        let spb = (sr as f64 * 60.0 / bpm) as usize; // samples per beat
        let mix_a = (spb * 8) as u64; // start fade at beat 8
        let mix_b = 0;

        let params = TransitionParams::new(
            4,
            CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear),
            mix_a,
            mix_b,
        ).unwrap();

        let result = CrossfadeEngine::render(
            &track_a, &track_b, &grid_a, &grid_b, &params, 4,
        ).unwrap();

        let buf = result.buffer();

        // Context before transition should be pure track A
        assert!((buf.samples()[0] - 1.0).abs() < 0.01, "pre-context should be track A");

        // After transition should be pure track B
        let last = buf.samples()[buf.num_frames() - 1];
        assert!((last - 0.5).abs() < 0.01, "post-context should be track B");

        // Transition region should exist
        assert!(result.transition_start_sample() > 0);
        assert!(result.transition_end_sample() > result.transition_start_sample());
    }

    #[test]
    fn render_stereo_crossfade() {
        let sr = 44100u32;
        let bpm = 128.0;
        let frames = sr as usize * 5;
        // Stereo: L=1.0, R=0.5 for track A; L=0.0, R=1.0 for track B
        let samples_a: Vec<f32> = (0..frames).flat_map(|_| vec![1.0, 0.5]).collect();
        let samples_b: Vec<f32> = (0..frames).flat_map(|_| vec![0.0, 1.0]).collect();
        let track_a = AudioBuffer::new(samples_a, sr, 2).unwrap();
        let track_b = AudioBuffer::new(samples_b, sr, 2).unwrap();
        let grid_a = simple_grid(bpm, sr, frames as u64);
        let grid_b = simple_grid(bpm, sr, frames as u64);

        let params = TransitionParams::new(
            4,
            CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear),
            0,
            0,
        ).unwrap();

        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        assert_eq!(result.buffer().channels(), 2);
        // First frame: L should be ~1.0 (track A), R should be ~0.5 (track A)
        let first_frame = result.buffer().get_frame(0).unwrap();
        assert!((first_frame[0] - 1.0).abs() < 0.01);
        assert!((first_frame[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn reject_mismatched_sample_rates() {
        let track_a = constant_buffer(1.0, 44100, 44100);
        let track_b = constant_buffer(1.0, 48000, 48000);
        let grid_a = simple_grid(120.0, 44100, 44100);
        let grid_b = simple_grid(120.0, 48000, 48000);

        let params = TransitionParams::default_16_beat(0, 0);
        let err = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap_err();

        assert!(matches!(err, CrossfadeError::SampleRateMismatch { .. }));
    }

    #[test]
    fn reject_mismatched_channels() {
        let track_a = AudioBuffer::new(vec![0.0; 44100], 44100, 1).unwrap();
        let track_b = AudioBuffer::new(vec![0.0; 44100 * 2], 44100, 2).unwrap();
        let grid_a = simple_grid(120.0, 44100, 44100);
        let grid_b = simple_grid(120.0, 44100, 44100);

        let params = TransitionParams::default_16_beat(0, 0);
        let err = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap_err();

        assert!(matches!(err, CrossfadeError::ChannelMismatch { .. }));
    }

    #[test]
    fn reject_track_too_short() {
        let track_a = constant_buffer(1.0, 1000, 44100);
        let track_b = constant_buffer(1.0, 44100 * 10, 44100);
        let grid_a = simple_grid(120.0, 44100, 1000);
        let grid_b = simple_grid(120.0, 44100, 44100 * 10);

        // 16-beat transition at 120 BPM needs ~8 seconds of audio
        let params = TransitionParams::default_16_beat(0, 0);
        let err = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap_err();

        assert!(matches!(err, CrossfadeError::TrackTooShort { track: "A", .. }));
    }

    #[test]
    fn find_mix_point_snaps_to_downbeat() {
        let grid = simple_grid(120.0, 44100, 44100 * 30);
        let spb = (44100.0 * 60.0 / 120.0) as u64; // 22050 samples per beat

        // Ask for a point near beat 5 — should snap to downbeat at beat 4 (index 4)
        let point = CrossfadeEngine::find_mix_point(&grid, spb * 5).unwrap();
        // Nearest downbeat is at beat 4 (0-indexed, downbeats at 0,4,8,...)
        assert_eq!(point, spb * 4);
    }

    #[test]
    fn output_sample_rate_matches_input() {
        let sr = 44100u32;
        let track_a = constant_buffer(1.0, sr as usize * 10, sr);
        let track_b = constant_buffer(0.5, sr as usize * 10, sr);
        let grid_a = simple_grid(120.0, sr, sr as u64 * 10);
        let grid_b = simple_grid(120.0, sr, sr as u64 * 10);

        let params = TransitionParams::default_16_beat(0, 0);
        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        assert_eq!(result.buffer().sample_rate(), 44100);
    }

    #[test]
    fn render_metadata_is_populated() {
        let sr = 44100u32;
        let track_a = constant_buffer(1.0, sr as usize * 10, sr);
        let track_b = constant_buffer(0.5, sr as usize * 10, sr);
        let grid_a = simple_grid(120.0, sr, sr as u64 * 10);
        let grid_b = simple_grid(128.0, sr, sr as u64 * 10);

        let params = TransitionParams::new(
            4,
            CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower),
            0,
            0,
        ).unwrap();

        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        let meta = result.metadata();
        assert!((meta.track_a_bpm - 120.0).abs() < 0.01);
        assert!((meta.track_b_bpm - 128.0).abs() < 0.01);
        assert_eq!(meta.curve_type, CrossfadeCurve::EqualPower);
        assert_eq!(meta.transition_beats, 4);
    }

    #[test]
    fn linear_crossfade_midpoint_is_average() {
        let sr = 44100u32;
        let track_a = constant_buffer(1.0, sr as usize * 10, sr);
        let track_b = constant_buffer(0.0, sr as usize * 10, sr);
        let grid_a = simple_grid(120.0, sr, sr as u64 * 10);
        let grid_b = simple_grid(120.0, sr, sr as u64 * 10);

        let params = TransitionParams::new(
            4,
            CrossfadeStyle::VolumeFade(CrossfadeCurve::Linear),
            0,
            0,
        ).unwrap();

        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        // At midpoint of linear crossfade: 0.5*1.0 + 0.5*0.0 = 0.5
        let mid = result.buffer().num_frames() / 2;
        let mid_val = result.buffer().samples()[mid];
        assert!(
            (mid_val - 0.5).abs() < 0.01,
            "midpoint should be 0.5, got {mid_val}"
        );
    }

    #[test]
    fn transition_16_beat_window() {
        let sr = 44100u32;
        let bpm = 120.0;
        let spb = (sr as f64 * 60.0 / bpm) as usize;
        let expected_frames = spb * 16;

        let track_a = constant_buffer(1.0, sr as usize * 30, sr);
        let track_b = constant_buffer(1.0, sr as usize * 30, sr);
        let grid_a = simple_grid(bpm, sr, sr as u64 * 30);
        let grid_b = simple_grid(bpm, sr, sr as u64 * 30);

        let params = TransitionParams::default_16_beat(0, 0);
        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        assert_eq!(result.buffer().num_frames(), expected_frames);
        assert_eq!(result.transition_duration_samples(), expected_frames as u64);
    }

    #[test]
    fn transition_32_beat_window() {
        let sr = 44100u32;
        let bpm = 128.0;
        let spb = (sr as f64 * 60.0 / bpm) as usize;
        let expected_frames = spb * 32;

        let track_a = constant_buffer(1.0, sr as usize * 60, sr);
        let track_b = constant_buffer(1.0, sr as usize * 60, sr);
        let grid_a = simple_grid(bpm, sr, sr as u64 * 60);
        let grid_b = simple_grid(bpm, sr, sr as u64 * 60);

        let params = TransitionParams::default_32_beat(0, 0);
        let result = CrossfadeEngine::render_transition_only(
            &track_a, &track_b, &grid_a, &grid_b, &params,
        ).unwrap();

        assert_eq!(result.buffer().num_frames(), expected_frames);
    }
}
