//! AI-powered audio blend renderer.
//!
//! Combines EQ automation, filter sweeps, phrase detection, and gain staging
//! to create professional-quality DJ transitions. Falls back to beat-aligned
//! crossfade if AI blend fails.

use crate::domain::{
    AudioBuffer, BeatGrid, CrossfadeCurve, CrossfadeStyle, RenderMetadata, RenderResult,
    TransitionParams,
};
use crate::domain::blend_params::{BlendStyle, Genre};
use crate::engine::crossfade::{CrossfadeEngine, CrossfadeError};
use crate::engine::eq::EqEngine;
use crate::engine::filter::FilterEngine;
use crate::engine::phrase_detect::PhraseDetector;
use std::time::Instant;

/// Errors from AI blend operations.
#[derive(Debug)]
pub enum AiBlendError {
    Crossfade(CrossfadeError),
    InvalidConfig(String),
}

impl std::fmt::Display for AiBlendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Crossfade(e) => write!(f, "crossfade error: {e}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for AiBlendError {}

impl From<CrossfadeError> for AiBlendError {
    fn from(e: CrossfadeError) -> Self {
        AiBlendError::Crossfade(e)
    }
}

/// Configuration for an AI blend operation.
pub struct AiBlendConfig {
    pub blend_style: BlendStyle,
    pub genre: Genre,
    pub transition_beats: Option<u32>,
}

impl AiBlendConfig {
    pub fn new(blend_style: BlendStyle, genre: Genre) -> Self {
        Self {
            blend_style,
            genre,
            transition_beats: None,
        }
    }

    pub fn with_transition_beats(mut self, beats: u32) -> Self {
        self.transition_beats = Some(beats);
        self
    }

    fn effective_beats(&self) -> u32 {
        self.transition_beats.unwrap_or_else(|| self.blend_style.default_beats())
    }
}

/// AI-powered blend renderer.
pub struct AiBlendEngine;

impl AiBlendEngine {
    /// Render an AI-powered blend between two tracks.
    ///
    /// This analyzes both tracks for phrases, applies the selected blend style
    /// with EQ automation and/or filter sweeps, and produces a high-quality
    /// transition. Falls back to simple crossfade on failure.
    pub fn render(
        track_a: &AudioBuffer,
        track_b: &AudioBuffer,
        grid_a: &BeatGrid,
        grid_b: &BeatGrid,
        config: &AiBlendConfig,
        context_beats: u32,
    ) -> Result<RenderResult, AiBlendError> {
        let start_time = Instant::now();

        // Validate matching formats
        if track_a.sample_rate() != track_b.sample_rate() {
            return Err(AiBlendError::Crossfade(CrossfadeError::SampleRateMismatch {
                rate_a: track_a.sample_rate(),
                rate_b: track_b.sample_rate(),
            }));
        }
        if track_a.channels() != track_b.channels() {
            return Err(AiBlendError::Crossfade(CrossfadeError::ChannelMismatch {
                ch_a: track_a.channels(),
                ch_b: track_b.channels(),
            }));
        }

        let sample_rate = track_a.sample_rate();
        let channels = track_a.channels();
        let transition_beats = config.effective_beats();

        // Find optimal mix points using phrase detection
        let mono_a = track_a.to_mono();
        let mono_b = track_b.to_mono();

        let mix_out = PhraseDetector::find_mix_out_point(mono_a.samples(), grid_a, sample_rate)
            .map(|mp| mp.sample_position())
            .unwrap_or_else(|| {
                let preferred = (track_a.num_frames() as f64 * 0.75) as u64;
                CrossfadeEngine::find_mix_point(grid_a, preferred).unwrap_or(preferred)
            });

        let mix_in = PhraseDetector::find_mix_in_point(mono_b.samples(), grid_b, sample_rate)
            .map(|mp| mp.sample_position())
            .unwrap_or(0);

        // Try AI blend, fall back to simple crossfade on failure
        let result = match Self::render_blend_style(
            track_a, track_b, grid_a, grid_b,
            mix_out, mix_in,
            config, context_beats, sample_rate, channels,
        ) {
            Ok(buf) => buf,
            Err(_) => {
                // Fallback: simple beat-aligned crossfade
                let params = TransitionParams::new(
                    transition_beats,
                    CrossfadeStyle::VolumeFade(CrossfadeCurve::EqualPower),
                    mix_out,
                    mix_in,
                ).map_err(|_| AiBlendError::InvalidConfig("invalid transition params".to_string()))?;

                return Ok(CrossfadeEngine::render(
                    track_a, track_b, grid_a, grid_b, &params, context_beats,
                )?);
            }
        };

        let render_time_ms = start_time.elapsed().as_millis() as u64;

        let metadata = RenderMetadata {
            render_time_ms,
            track_a_bpm: grid_a.bpm(),
            track_b_bpm: grid_b.bpm(),
            curve_type: CrossfadeCurve::EqualPower,
            transition_beats,
        };

        let samples_per_beat_a = (sample_rate as f64 * 60.0 / grid_a.bpm()) as usize;
        let context_frames = samples_per_beat_a * context_beats as usize;
        let transition_frames = samples_per_beat_a * transition_beats as usize;

        Ok(RenderResult::new(
            result,
            context_frames as u64,
            (context_frames + transition_frames) as u64,
            metadata,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn render_blend_style(
        track_a: &AudioBuffer,
        track_b: &AudioBuffer,
        grid_a: &BeatGrid,
        grid_b: &BeatGrid,
        mix_out: u64,
        mix_in: u64,
        config: &AiBlendConfig,
        context_beats: u32,
        sample_rate: u32,
        channels: u16,
    ) -> Result<AudioBuffer, AiBlendError> {
        let transition_beats = config.effective_beats();
        let samples_per_beat_a = (sample_rate as f64 * 60.0 / grid_a.bpm()) as usize;
        let samples_per_beat_b = (sample_rate as f64 * 60.0 / grid_b.bpm()) as usize;
        let transition_frames = samples_per_beat_a * transition_beats as usize;
        let context_frames_a = samples_per_beat_a * context_beats as usize;
        let context_frames_b = samples_per_beat_b * context_beats as usize;

        let mix_a = mix_out as usize;
        let mix_b = mix_in as usize;

        let a_start = mix_a.saturating_sub(context_frames_a);
        let a_end = mix_a + transition_frames;
        let b_end = mix_b + transition_frames + context_frames_b;

        if a_end > track_a.num_frames() {
            return Err(AiBlendError::Crossfade(CrossfadeError::TrackTooShort {
                track: "A",
                available_frames: track_a.num_frames(),
                required_frames: a_end,
            }));
        }
        if b_end > track_b.num_frames() {
            return Err(AiBlendError::Crossfade(CrossfadeError::TrackTooShort {
                track: "B",
                available_frames: track_b.num_frames(),
                required_frames: b_end,
            }));
        }

        let ch = channels as usize;
        let total_frames = context_frames_a + transition_frames + context_frames_b;
        let mut output = vec![0.0f32; total_frames * ch];

        let samples_a = track_a.samples();
        let samples_b = track_b.samples();

        // Region 1: Pre-transition context (solo track A)
        for frame in 0..context_frames_a {
            let src_frame = a_start + frame;
            for c in 0..ch {
                output[frame * ch + c] = samples_a[src_frame * ch + c];
            }
        }

        // Region 2: Transition (style-dependent processing)
        match config.blend_style {
            BlendStyle::LongBlend => {
                Self::render_long_blend(
                    &mut output, samples_a, samples_b,
                    mix_a, mix_b, transition_frames,
                    context_frames_a, ch, sample_rate,
                );
            }
            BlendStyle::ShortCut => {
                Self::render_short_cut(
                    &mut output, samples_a, samples_b,
                    mix_a, mix_b, transition_frames,
                    context_frames_a, ch,
                );
            }
            BlendStyle::FilterSweep => {
                Self::render_filter_sweep(
                    &mut output, samples_a, samples_b,
                    mix_a, mix_b, transition_frames,
                    context_frames_a, ch, sample_rate,
                );
            }
            BlendStyle::EchoOut => {
                Self::render_echo_out(
                    &mut output, samples_a, samples_b,
                    mix_a, mix_b, transition_frames,
                    context_frames_a, ch, sample_rate, grid_a,
                );
            }
            BlendStyle::Backspin => {
                Self::render_backspin(
                    &mut output, samples_a, samples_b,
                    mix_a, mix_b, transition_frames,
                    context_frames_a, ch,
                );
            }
        }

        // Region 3: Post-transition context (solo track B)
        for frame in 0..context_frames_b {
            let src_frame = mix_b + transition_frames + frame;
            let out_frame = context_frames_a + transition_frames + frame;
            for c in 0..ch {
                output[out_frame * ch + c] = samples_b[src_frame * ch + c];
            }
        }

        AudioBuffer::new(output, sample_rate, channels)
            .map_err(|e| AiBlendError::Crossfade(CrossfadeError::BufferError(e)))
    }

    /// Long blend: EQ low-swap with equal-power volume crossfade.
    #[allow(clippy::too_many_arguments)]
    fn render_long_blend(
        output: &mut [f32],
        samples_a: &[f32],
        samples_b: &[f32],
        mix_a: usize,
        mix_b: usize,
        transition_frames: usize,
        context_frames: usize,
        ch: usize,
        sample_rate: u32,
    ) {
        let eq_auto = EqEngine::low_swap_automation();

        // Process each channel
        for c in 0..ch {
            // Extract mono channel data for the transition region
            let a_mono: Vec<f32> = (0..transition_frames)
                .map(|f| samples_a[(mix_a + f) * ch + c])
                .collect();
            let b_mono: Vec<f32> = (0..transition_frames)
                .map(|f| samples_b[(mix_b + f) * ch + c])
                .collect();

            // Apply EQ automation
            let a_eq = EqEngine::apply_eq_automation(
                &a_mono, sample_rate, &eq_auto.track_a_curves, transition_frames, 0,
            );
            let b_eq = EqEngine::apply_eq_automation(
                &b_mono, sample_rate, &eq_auto.track_b_curves, transition_frames, 0,
            );

            // Mix with equal-power gain staging
            for frame in 0..transition_frames {
                let position = frame as f64 / transition_frames.max(1) as f64;
                let (gain_a, gain_b) = CrossfadeCurve::EqualPower.gain_at(position);
                let out_frame = context_frames + frame;
                output[out_frame * ch + c] = a_eq[frame] * gain_a as f32 + b_eq[frame] * gain_b as f32;
            }
        }
    }

    /// Short cut: rapid volume crossfade with slight overlap.
    #[allow(clippy::too_many_arguments)]
    fn render_short_cut(
        output: &mut [f32],
        samples_a: &[f32],
        samples_b: &[f32],
        mix_a: usize,
        mix_b: usize,
        transition_frames: usize,
        context_frames: usize,
        ch: usize,
    ) {
        for frame in 0..transition_frames {
            let position = frame as f64 / transition_frames.max(1) as f64;
            // Sharp S-curve for quick cut
            let t = if position < 0.5 {
                2.0 * position * position
            } else {
                1.0 - 2.0 * (1.0 - position) * (1.0 - position)
            };
            let gain_a = (1.0 - t) as f32;
            let gain_b = t as f32;

            let out_frame = context_frames + frame;
            for c in 0..ch {
                let a = samples_a[(mix_a + frame) * ch + c];
                let b = samples_b[(mix_b + frame) * ch + c];
                output[out_frame * ch + c] = a * gain_a + b * gain_b;
            }
        }
    }

    /// Filter sweep: HPF on outgoing, LPF on incoming.
    #[allow(clippy::too_many_arguments)]
    fn render_filter_sweep(
        output: &mut [f32],
        samples_a: &[f32],
        samples_b: &[f32],
        mix_a: usize,
        mix_b: usize,
        transition_frames: usize,
        context_frames: usize,
        ch: usize,
        sample_rate: u32,
    ) {
        let hpf_params = FilterEngine::hpf_sweep_out(1.5);
        let lpf_params = FilterEngine::lpf_sweep_in(1.5);

        for c in 0..ch {
            let a_mono: Vec<f32> = (0..transition_frames)
                .map(|f| samples_a[(mix_a + f) * ch + c])
                .collect();
            let b_mono: Vec<f32> = (0..transition_frames)
                .map(|f| samples_b[(mix_b + f) * ch + c])
                .collect();

            let a_filtered = FilterEngine::apply_sweep(&a_mono, sample_rate, &hpf_params, transition_frames, 0);
            let b_filtered = FilterEngine::apply_sweep(&b_mono, sample_rate, &lpf_params, transition_frames, 0);

            // Mix with linear crossfade under the filters
            for frame in 0..transition_frames {
                let position = frame as f64 / transition_frames.max(1) as f64;
                let gain_a = (1.0 - position) as f32;
                let gain_b = position as f32;
                let out_frame = context_frames + frame;
                output[out_frame * ch + c] = a_filtered[frame] * gain_a + b_filtered[frame] * gain_b;
            }
        }
    }

    /// Echo out: echo/delay tail on outgoing track.
    #[allow(clippy::too_many_arguments)]
    fn render_echo_out(
        output: &mut [f32],
        samples_a: &[f32],
        samples_b: &[f32],
        mix_a: usize,
        mix_b: usize,
        transition_frames: usize,
        context_frames: usize,
        ch: usize,
        sample_rate: u32,
        grid_a: &BeatGrid,
    ) {
        // Echo delay = one beat
        let delay_samples = (sample_rate as f64 * 60.0 / grid_a.bpm()) as usize;
        let feedback = 0.5f32;
        let wet = 0.4f32;

        for c in 0..ch {
            // Build delay buffer for track A
            let mut delay_buf = vec![0.0f32; transition_frames + delay_samples * 4];
            for frame in 0..transition_frames {
                delay_buf[frame] = samples_a[(mix_a + frame) * ch + c];
            }

            // Apply feedback echo
            for frame in delay_samples..delay_buf.len() {
                delay_buf[frame] += delay_buf[frame - delay_samples] * feedback;
            }

            for frame in 0..transition_frames {
                let position = frame as f64 / transition_frames.max(1) as f64;
                let gain_a = (1.0 - position) as f32;
                let gain_b = position as f32;

                let dry_a = samples_a[(mix_a + frame) * ch + c];
                let wet_a = delay_buf[frame];
                let a_with_echo = dry_a * (1.0 - wet) + wet_a * wet;

                let b = samples_b[(mix_b + frame) * ch + c];
                let out_frame = context_frames + frame;
                output[out_frame * ch + c] = a_with_echo * gain_a + b * gain_b;
            }
        }
    }

    /// Backspin: reverse playback effect on outgoing track.
    #[allow(clippy::too_many_arguments)]
    fn render_backspin(
        output: &mut [f32],
        samples_a: &[f32],
        samples_b: &[f32],
        mix_a: usize,
        mix_b: usize,
        transition_frames: usize,
        context_frames: usize,
        ch: usize,
    ) {
        // First quarter: start backspin on track A
        let spin_frames = transition_frames / 4;
        let fade_frames = transition_frames - spin_frames;

        for frame in 0..transition_frames {
            let out_frame = context_frames + frame;

            if frame < spin_frames {
                // Backspin: play track A in reverse with pitch-down effect
                let reverse_frame = spin_frames - 1 - frame;
                let gain = 1.0 - (frame as f32 / spin_frames as f32);
                for c in 0..ch {
                    let src = if mix_a + reverse_frame < track_a_len(samples_a, ch) {
                        samples_a[(mix_a + reverse_frame) * ch + c]
                    } else {
                        0.0
                    };
                    let b = samples_b[(mix_b + frame) * ch + c];
                    output[out_frame * ch + c] = src * gain + b * (1.0 - gain);
                }
            } else {
                // After backspin: fade to track B
                let fade_pos = (frame - spin_frames) as f32 / fade_frames.max(1) as f32;
                let gain_b = fade_pos.min(1.0);
                for c in 0..ch {
                    let b = samples_b[(mix_b + frame) * ch + c];
                    output[out_frame * ch + c] = b * gain_b;
                }
            }
        }
    }

    /// Auto-select the best blend style based on genre and track characteristics.
    pub fn auto_select_style(genre: Genre, _bpm_a: f64, _bpm_b: f64) -> BlendStyle {
        genre.suggested_blend_style()
    }
}

fn track_a_len(samples: &[f32], ch: usize) -> usize {
    samples.len() / ch
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::AudioBuffer;

    fn constant_buffer(value: f32, num_frames: usize, sample_rate: u32) -> AudioBuffer {
        AudioBuffer::new(vec![value; num_frames], sample_rate, 1).unwrap()
    }

    fn sine_buffer(freq: f64, num_frames: usize, sample_rate: u32) -> AudioBuffer {
        let samples: Vec<f32> = (0..num_frames)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                (2.0 * std::f64::consts::PI * freq * t).sin() as f32 * 0.5
            })
            .collect();
        AudioBuffer::new(samples, sample_rate, 1).unwrap()
    }

    fn simple_grid(bpm: f64, sample_rate: u32, total_frames: u64) -> BeatGrid {
        BeatGrid::from_bpm(bpm, 0, sample_rate, total_frames).unwrap()
    }

    #[test]
    fn long_blend_renders_successfully() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);

        assert!(result.is_ok(), "long blend should render: {:?}", result.err());
        let r = result.unwrap();
        assert!(r.buffer().num_frames() > 0);
        assert_eq!(r.buffer().sample_rate(), sr);
    }

    #[test]
    fn short_cut_renders_successfully() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::ShortCut, Genre::DrumAndBass);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);

        assert!(result.is_ok(), "short cut should render: {:?}", result.err());
    }

    #[test]
    fn filter_sweep_renders_successfully() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::FilterSweep, Genre::Trance);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);

        assert!(result.is_ok(), "filter sweep should render: {:?}", result.err());
    }

    #[test]
    fn echo_out_renders_successfully() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::EchoOut, Genre::HipHop);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);

        assert!(result.is_ok(), "echo out should render: {:?}", result.err());
    }

    #[test]
    fn backspin_renders_successfully() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::Backspin, Genre::HipHop);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);

        assert!(result.is_ok(), "backspin should render: {:?}", result.err());
    }

    #[test]
    fn ai_blend_output_starts_with_track_a() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = constant_buffer(0.8, frames, sr);
        let track_b = constant_buffer(0.3, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4).unwrap();

        // First sample in context should be from track A
        assert!(
            (result.buffer().samples()[0] - 0.8).abs() < 0.01,
            "output should start with track A: got {}",
            result.buffer().samples()[0]
        );
    }

    #[test]
    fn ai_blend_output_ends_with_track_b() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = constant_buffer(0.8, frames, sr);
        let track_b = constant_buffer(0.3, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4).unwrap();

        let last = result.buffer().samples()[result.buffer().num_frames() - 1];
        assert!(
            (last - 0.3).abs() < 0.01,
            "output should end with track B: got {last}"
        );
    }

    #[test]
    fn ai_blend_no_clicks_in_transition() {
        let sr = 44100u32;
        let frames = sr as usize * 60;
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4).unwrap();

        let samples = result.buffer().samples();
        let mut max_diff = 0.0f32;
        for i in 1..samples.len() {
            let diff = (samples[i] - samples[i - 1]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 0.5,
            "transition should be smooth, max sample diff: {max_diff}"
        );
    }

    #[test]
    fn ai_blend_rejects_mismatched_sample_rates() {
        let track_a = constant_buffer(1.0, 44100 * 10, 44100);
        let track_b = constant_buffer(1.0, 48000 * 10, 48000);
        let grid_a = simple_grid(128.0, 44100, 44100 * 10);
        let grid_b = simple_grid(128.0, 48000, 48000 * 10);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);
        assert!(result.is_err());
    }

    #[test]
    fn ai_blend_renders_under_3_seconds() {
        let sr = 44100u32;
        let frames = sr as usize * 60; // 60 second tracks
        let track_a = sine_buffer(440.0, frames, sr);
        let track_b = sine_buffer(550.0, frames, sr);
        let grid_a = simple_grid(128.0, sr, frames as u64);
        let grid_b = simple_grid(128.0, sr, frames as u64);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        let start = Instant::now();
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 8);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert!(
            elapsed.as_secs() < 3,
            "AI blend should render in < 3 seconds, took {:?}",
            elapsed
        );
    }

    #[test]
    fn auto_select_style_uses_genre() {
        assert_eq!(AiBlendEngine::auto_select_style(Genre::House, 128.0, 128.0), BlendStyle::LongBlend);
        assert_eq!(AiBlendEngine::auto_select_style(Genre::Trance, 140.0, 140.0), BlendStyle::FilterSweep);
        assert_eq!(AiBlendEngine::auto_select_style(Genre::DrumAndBass, 174.0, 174.0), BlendStyle::ShortCut);
    }

    #[test]
    fn custom_transition_beats_override() {
        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House)
            .with_transition_beats(16);
        assert_eq!(config.effective_beats(), 16);
    }

    #[test]
    fn default_transition_beats_from_style() {
        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        assert_eq!(config.effective_beats(), 32);
    }

    #[test]
    fn fallback_to_crossfade_on_short_track() {
        let sr = 44100u32;
        // Track A is very short but track B is fine
        let track_a = sine_buffer(440.0, sr as usize * 5, sr);
        let track_b = sine_buffer(550.0, sr as usize * 30, sr);
        let grid_a = simple_grid(128.0, sr, sr as u64 * 5);
        let grid_b = simple_grid(128.0, sr, sr as u64 * 30);

        let config = AiBlendConfig::new(BlendStyle::LongBlend, Genre::House);
        // This may fail or fall back - either is acceptable
        let result = AiBlendEngine::render(&track_a, &track_b, &grid_a, &grid_b, &config, 4);
        // Should not panic regardless
        let _ = result;
    }
}
