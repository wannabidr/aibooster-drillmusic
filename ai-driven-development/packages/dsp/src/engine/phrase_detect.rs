//! Phrase-aware mix point detection.
//!
//! Analyzes audio energy patterns to detect intro/outro phrases
//! and select optimal mix-in and mix-out points.

use crate::domain::beat_grid::BeatGrid;
use crate::domain::blend_params::{Phrase, PhraseType};
use crate::domain::mix_point::{MixPoint, MixType};

/// Standard phrase lengths in beats for electronic music.
#[cfg(test)]
const PHRASE_LENGTHS: &[usize] = &[32, 16, 8];

/// Phrase detection engine.
pub struct PhraseDetector;

impl PhraseDetector {
    /// Detect phrases in an audio track by analyzing energy per bar.
    ///
    /// # Arguments
    /// * `samples` - mono audio samples
    /// * `grid` - beat grid for the track
    /// * `sample_rate` - sample rate in Hz
    pub fn detect_phrases(
        samples: &[f32],
        grid: &BeatGrid,
        sample_rate: u32,
    ) -> Vec<Phrase> {
        if grid.beat_count() < 8 {
            return Vec::new();
        }

        let bar_energies = Self::compute_bar_energies(samples, grid, sample_rate);
        if bar_energies.is_empty() {
            return Vec::new();
        }

        let beats_per_bar = grid.beats_per_bar() as usize;
        let phrase_len_bars = Self::detect_phrase_length(&bar_energies);
        let _phrase_len_beats = phrase_len_bars * beats_per_bar;

        let mut phrases = Vec::new();
        let total_bars = bar_energies.len();
        let mut bar_idx = 0;

        while bar_idx + phrase_len_bars <= total_bars {
            let start_beat = bar_idx * beats_per_bar;
            let end_beat = (bar_idx + phrase_len_bars) * beats_per_bar;
            let end_beat = end_beat.min(grid.beat_count());

            let avg_energy: f64 = bar_energies[bar_idx..bar_idx + phrase_len_bars]
                .iter()
                .sum::<f64>()
                / phrase_len_bars as f64;

            let phrase_type = Self::classify_phrase(
                bar_idx,
                total_bars,
                phrase_len_bars,
                avg_energy,
                &bar_energies,
            );

            phrases.push(Phrase {
                start_beat,
                end_beat,
                energy: avg_energy,
                phrase_type,
            });

            bar_idx += phrase_len_bars;
        }

        // Handle remaining bars
        if bar_idx < total_bars {
            let start_beat = bar_idx * beats_per_bar;
            let end_beat = grid.beat_count();
            let remaining = total_bars - bar_idx;
            let avg_energy: f64 = bar_energies[bar_idx..].iter().sum::<f64>() / remaining as f64;

            phrases.push(Phrase {
                start_beat,
                end_beat,
                energy: avg_energy,
                phrase_type: PhraseType::Body,
            });
        }

        phrases
    }

    /// Compute the RMS energy for each bar.
    fn compute_bar_energies(samples: &[f32], grid: &BeatGrid, _sample_rate: u32) -> Vec<f64> {
        let beats_per_bar = grid.beats_per_bar() as usize;
        let positions = grid.beat_positions();
        let num_bars = grid.bar_count();
        let mut energies = Vec::with_capacity(num_bars);

        for bar in 0..num_bars {
            let start_beat = bar * beats_per_bar;
            let end_beat = start_beat + beats_per_bar;

            let start_sample = positions[start_beat] as usize;
            let end_sample = if end_beat < positions.len() {
                positions[end_beat] as usize
            } else {
                samples.len()
            };
            let end_sample = end_sample.min(samples.len());

            if start_sample >= end_sample {
                energies.push(0.0);
                continue;
            }

            let segment = &samples[start_sample..end_sample];
            let rms = (segment.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>()
                / segment.len() as f64)
                .sqrt();
            energies.push(rms);
        }

        // Normalize to 0.0..1.0
        let max_energy = energies.iter().cloned().fold(0.0f64, f64::max);
        if max_energy > 0.0 {
            for e in &mut energies {
                *e /= max_energy;
            }
        }

        energies
    }

    /// Detect the dominant phrase length by looking at energy pattern repetition.
    fn detect_phrase_length(bar_energies: &[f64]) -> usize {
        let total = bar_energies.len();

        // Try common phrase lengths (in bars): 8, 4, 2
        for &phrase_bars in &[8usize, 4, 2] {
            if total < phrase_bars * 2 {
                continue;
            }

            // Check if energy pattern repeats at this phrase length
            let mut correlation = 0.0;
            let mut count = 0;
            for i in 0..total.saturating_sub(phrase_bars) {
                let diff = (bar_energies[i] - bar_energies[i + phrase_bars]).abs();
                correlation += 1.0 - diff;
                count += 1;
            }
            if count > 0 {
                correlation /= count as f64;
            }

            // If pattern repeats with >60% correlation, use this length
            if correlation > 0.6 {
                return phrase_bars;
            }
        }

        // Default to 4-bar phrases (16 beats in 4/4)
        4
    }

    /// Classify a phrase segment based on position and energy.
    fn classify_phrase(
        bar_idx: usize,
        total_bars: usize,
        phrase_len_bars: usize,
        avg_energy: f64,
        bar_energies: &[f64],
    ) -> PhraseType {
        let is_first = bar_idx == 0;
        let is_last = bar_idx + phrase_len_bars >= total_bars;

        // Check for energy ramp (intro/outro pattern)
        let segment = &bar_energies[bar_idx..bar_idx + phrase_len_bars];
        let first_half_energy: f64 = segment[..segment.len() / 2].iter().sum::<f64>()
            / (segment.len() / 2) as f64;
        let second_half_energy: f64 = segment[segment.len() / 2..].iter().sum::<f64>()
            / (segment.len() - segment.len() / 2) as f64;

        if is_first && avg_energy < 0.4 {
            PhraseType::Intro
        } else if is_last && avg_energy < 0.4 {
            PhraseType::Outro
        } else if is_first && second_half_energy > first_half_energy * 1.3 {
            PhraseType::Intro
        } else if is_last && first_half_energy > second_half_energy * 1.3 {
            PhraseType::Outro
        } else if avg_energy > 0.7 && second_half_energy > first_half_energy {
            PhraseType::Drop
        } else if avg_energy < 0.3 {
            PhraseType::Breakdown
        } else if second_half_energy > first_half_energy * 1.5 {
            PhraseType::Buildup
        } else {
            PhraseType::Body
        }
    }

    /// Find the optimal mix-out point in a track (for the outgoing track).
    /// Prefers the start of an outro phrase, or the last low-energy section.
    pub fn find_mix_out_point(
        samples: &[f32],
        grid: &BeatGrid,
        sample_rate: u32,
    ) -> Option<MixPoint> {
        let phrases = Self::detect_phrases(samples, grid, sample_rate);
        if phrases.is_empty() {
            return None;
        }

        // Look for outro phrase
        for phrase in phrases.iter().rev() {
            if phrase.phrase_type == PhraseType::Outro {
                let beat_idx = phrase.start_beat;
                if beat_idx < grid.beat_positions().len() {
                    let sample_pos = grid.beat_positions()[beat_idx];
                    return Some(MixPoint::new(sample_pos, beat_idx, 0.9, MixType::Outro).unwrap());
                }
            }
        }

        // Fallback: look for low-energy phrase near the end (last 25%)
        let threshold_idx = phrases.len() * 3 / 4;
        for phrase in phrases[threshold_idx..].iter() {
            if phrase.energy < 0.5 {
                let beat_idx = phrase.start_beat;
                if beat_idx < grid.beat_positions().len() {
                    let sample_pos = grid.beat_positions()[beat_idx];
                    return Some(MixPoint::new(sample_pos, beat_idx, 0.6, MixType::Outro).unwrap());
                }
            }
        }

        // Final fallback: 75% through the track
        let beat_idx = grid.beat_count() * 3 / 4;
        let beat_idx = beat_idx.min(grid.beat_count().saturating_sub(1));
        if beat_idx < grid.beat_positions().len() {
            let sample_pos = grid.beat_positions()[beat_idx];
            Some(MixPoint::new(sample_pos, beat_idx, 0.3, MixType::Outro).unwrap())
        } else {
            None
        }
    }

    /// Find the optimal mix-in point in a track (for the incoming track).
    /// Prefers the end of an intro phrase, or the first low-energy section.
    pub fn find_mix_in_point(
        samples: &[f32],
        grid: &BeatGrid,
        sample_rate: u32,
    ) -> Option<MixPoint> {
        let phrases = Self::detect_phrases(samples, grid, sample_rate);
        if phrases.is_empty() {
            return None;
        }

        // Look for intro phrase
        for phrase in &phrases {
            if phrase.phrase_type == PhraseType::Intro {
                let beat_idx = phrase.start_beat;
                if beat_idx < grid.beat_positions().len() {
                    let sample_pos = grid.beat_positions()[beat_idx];
                    return Some(MixPoint::new(sample_pos, beat_idx, 0.9, MixType::Intro).unwrap());
                }
            }
        }

        // Fallback: first beat
        if !grid.beat_positions().is_empty() {
            let sample_pos = grid.beat_positions()[0];
            Some(MixPoint::new(sample_pos, 0, 0.5, MixType::Intro).unwrap())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(bpm: f64, sample_rate: u32, duration_secs: f64) -> BeatGrid {
        let total_frames = (sample_rate as f64 * duration_secs) as u64;
        BeatGrid::from_bpm(bpm, 0, sample_rate, total_frames).unwrap()
    }

    /// Create audio with intro (quiet), body (loud), outro (quiet).
    fn create_structured_audio(sample_rate: u32, duration_secs: f64) -> Vec<f32> {
        let total = (sample_rate as f64 * duration_secs) as usize;
        let intro_end = total / 4;
        let outro_start = total * 3 / 4;

        (0..total)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                let base = (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32;
                if i < intro_end {
                    // Intro: ramp up from 0.1 to 0.8
                    let gain = 0.1 + 0.7 * (i as f32 / intro_end as f32);
                    base * gain
                } else if i > outro_start {
                    // Outro: ramp down from 0.8 to 0.1
                    let progress = (i - outro_start) as f32 / (total - outro_start) as f32;
                    let gain = 0.8 - 0.7 * progress;
                    base * gain
                } else {
                    // Body: full volume
                    base * 0.8
                }
            })
            .collect()
    }

    #[test]
    fn detects_phrases_in_structured_audio() {
        let sr = 44100;
        let duration = 60.0;
        let audio = create_structured_audio(sr, duration);
        let grid = make_grid(128.0, sr, duration);

        let phrases = PhraseDetector::detect_phrases(&audio, &grid, sr);
        assert!(!phrases.is_empty(), "should detect at least one phrase");

        // First phrase should be intro
        assert_eq!(phrases[0].phrase_type, PhraseType::Intro);
        // Last phrase should be outro or low-energy
        let last = phrases.last().unwrap();
        assert!(
            last.phrase_type == PhraseType::Outro || last.energy < 0.5,
            "last phrase should be outro or low energy, got {:?} with energy {}",
            last.phrase_type, last.energy
        );
    }

    #[test]
    fn phrase_lengths_are_standard() {
        let sr = 44100;
        let audio = create_structured_audio(sr, 60.0);
        let grid = make_grid(128.0, sr, 60.0);

        let phrases = PhraseDetector::detect_phrases(&audio, &grid, sr);
        for phrase in &phrases {
            let len = phrase.length_beats();
            assert!(
                PHRASE_LENGTHS.contains(&len) || len < PHRASE_LENGTHS[2],
                "phrase length {} beats is non-standard",
                len
            );
        }
    }

    #[test]
    fn intro_phrase_has_low_energy() {
        let sr = 44100;
        let audio = create_structured_audio(sr, 60.0);
        let grid = make_grid(128.0, sr, 60.0);

        let phrases = PhraseDetector::detect_phrases(&audio, &grid, sr);
        let intro = phrases.iter().find(|p| p.phrase_type == PhraseType::Intro);
        assert!(intro.is_some(), "should detect intro phrase");
        assert!(intro.unwrap().energy < 0.6, "intro should have lower energy");
    }

    #[test]
    fn finds_mix_out_point() {
        let sr = 44100;
        let audio = create_structured_audio(sr, 60.0);
        let grid = make_grid(128.0, sr, 60.0);

        let mix_out = PhraseDetector::find_mix_out_point(&audio, &grid, sr);
        assert!(mix_out.is_some(), "should find a mix-out point");

        let mp = mix_out.unwrap();
        assert_eq!(mp.mix_type(), MixType::Outro);
        assert!(mp.confidence() > 0.0);
        // Mix out should be in the latter portion of the track
        let total_samples = audio.len() as u64;
        assert!(
            mp.sample_position() > total_samples / 2,
            "mix-out should be in the second half of the track"
        );
    }

    #[test]
    fn finds_mix_in_point() {
        let sr = 44100;
        let audio = create_structured_audio(sr, 60.0);
        let grid = make_grid(128.0, sr, 60.0);

        let mix_in = PhraseDetector::find_mix_in_point(&audio, &grid, sr);
        assert!(mix_in.is_some(), "should find a mix-in point");

        let mp = mix_in.unwrap();
        assert_eq!(mp.mix_type(), MixType::Intro);
    }

    #[test]
    fn handles_short_track_gracefully() {
        let sr = 44100;
        let short_audio: Vec<f32> = vec![0.5; sr as usize]; // 1 second
        let grid = make_grid(128.0, sr, 1.0);

        let phrases = PhraseDetector::detect_phrases(&short_audio, &grid, sr);
        // Should not panic, may return empty or single phrase
        assert!(phrases.len() <= 2);
    }

    #[test]
    fn handles_silence_gracefully() {
        let sr = 44100;
        let silence: Vec<f32> = vec![0.0; sr as usize * 30];
        let grid = make_grid(128.0, sr, 30.0);

        let phrases = PhraseDetector::detect_phrases(&silence, &grid, sr);
        // All phrases should have zero energy
        for phrase in &phrases {
            assert!(
                phrase.energy < 0.01,
                "silence should have zero energy, got {}",
                phrase.energy
            );
        }
    }

    #[test]
    fn constant_energy_produces_body_phrases() {
        let sr = 44100;
        let constant: Vec<f32> = (0..sr as usize * 30)
            .map(|i| {
                let t = i as f64 / sr as f64;
                (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5
            })
            .collect();
        let grid = make_grid(128.0, sr, 30.0);

        let phrases = PhraseDetector::detect_phrases(&constant, &grid, sr);
        // With constant energy, most phrases should be Body or Drop (high energy)
        let body_or_drop: usize = phrases
            .iter()
            .filter(|p| p.phrase_type == PhraseType::Body || p.phrase_type == PhraseType::Drop)
            .count();
        assert!(
            body_or_drop as f64 / phrases.len() as f64 > 0.5,
            "constant energy should produce mostly body/drop phrases"
        );
    }
}
