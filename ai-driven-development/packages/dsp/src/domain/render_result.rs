use super::audio_buffer::AudioBuffer;
use super::crossfade::CrossfadeCurve;

/// Output of a crossfade render operation.
#[derive(Debug, Clone)]
pub struct RenderResult {
    buffer: AudioBuffer,
    transition_start_sample: u64,
    transition_end_sample: u64,
    metadata: RenderMetadata,
}

impl RenderResult {
    pub fn new(
        buffer: AudioBuffer,
        transition_start_sample: u64,
        transition_end_sample: u64,
        metadata: RenderMetadata,
    ) -> Self {
        Self {
            buffer,
            transition_start_sample,
            transition_end_sample,
            metadata,
        }
    }

    pub fn buffer(&self) -> &AudioBuffer {
        &self.buffer
    }

    pub fn into_buffer(self) -> AudioBuffer {
        self.buffer
    }

    pub fn transition_start_sample(&self) -> u64 {
        self.transition_start_sample
    }

    pub fn transition_end_sample(&self) -> u64 {
        self.transition_end_sample
    }

    pub fn metadata(&self) -> &RenderMetadata {
        &self.metadata
    }

    pub fn transition_duration_samples(&self) -> u64 {
        self.transition_end_sample.saturating_sub(self.transition_start_sample)
    }
}

/// Metadata about a completed render operation.
#[derive(Debug, Clone)]
pub struct RenderMetadata {
    pub render_time_ms: u64,
    pub track_a_bpm: f64,
    pub track_b_bpm: f64,
    pub curve_type: CrossfadeCurve,
    pub transition_beats: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_result() -> RenderResult {
        let buffer = AudioBuffer::new(vec![0.0; 44100], 44100, 1).unwrap();
        let metadata = RenderMetadata {
            render_time_ms: 150,
            track_a_bpm: 128.0,
            track_b_bpm: 126.0,
            curve_type: CrossfadeCurve::EqualPower,
            transition_beats: 16,
        };
        RenderResult::new(buffer, 10000, 30000, metadata)
    }

    #[test]
    fn render_result_accessors() {
        let result = make_test_result();
        assert_eq!(result.transition_start_sample(), 10000);
        assert_eq!(result.transition_end_sample(), 30000);
        assert_eq!(result.transition_duration_samples(), 20000);
        assert_eq!(result.buffer().sample_rate(), 44100);
    }

    #[test]
    fn render_metadata_fields() {
        let result = make_test_result();
        let meta = result.metadata();
        assert_eq!(meta.render_time_ms, 150);
        assert!((meta.track_a_bpm - 128.0).abs() < 1e-10);
        assert!((meta.track_b_bpm - 126.0).abs() < 1e-10);
        assert_eq!(meta.curve_type, CrossfadeCurve::EqualPower);
        assert_eq!(meta.transition_beats, 16);
    }

    #[test]
    fn into_buffer_ownership() {
        let result = make_test_result();
        let buf = result.into_buffer();
        assert_eq!(buf.num_frames(), 44100);
    }

    #[test]
    fn transition_duration_zero_when_equal() {
        let buffer = AudioBuffer::new(vec![0.0; 100], 44100, 1).unwrap();
        let metadata = RenderMetadata {
            render_time_ms: 0,
            track_a_bpm: 128.0,
            track_b_bpm: 128.0,
            curve_type: CrossfadeCurve::Linear,
            transition_beats: 16,
        };
        let result = RenderResult::new(buffer, 500, 500, metadata);
        assert_eq!(result.transition_duration_samples(), 0);
    }
}
