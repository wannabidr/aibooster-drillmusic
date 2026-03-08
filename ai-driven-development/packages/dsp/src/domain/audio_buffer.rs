/// Decoded audio data ready for DSP processing.
/// Domain type with zero external dependencies.
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

impl AudioBuffer {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Result<Self, AudioBufferError> {
        if channels == 0 {
            return Err(AudioBufferError::InvalidChannels);
        }
        if sample_rate == 0 {
            return Err(AudioBufferError::InvalidSampleRate);
        }
        if !samples.is_empty() && !samples.len().is_multiple_of(channels as usize) {
            return Err(AudioBufferError::SampleCountMismatch {
                samples: samples.len(),
                channels,
            });
        }
        Ok(Self {
            samples,
            sample_rate,
            channels,
        })
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.channels
    }

    pub fn num_frames(&self) -> usize {
        if self.channels == 0 {
            return 0;
        }
        self.samples.len() / self.channels as usize
    }

    pub fn duration_secs(&self) -> f64 {
        self.num_frames() as f64 / self.sample_rate as f64
    }

    /// Returns a single frame (one sample per channel) at the given index.
    pub fn get_frame(&self, frame_idx: usize) -> Option<&[f32]> {
        let ch = self.channels as usize;
        let start = frame_idx * ch;
        let end = start + ch;
        if end <= self.samples.len() {
            Some(&self.samples[start..end])
        } else {
            None
        }
    }

    /// Slice the buffer by frame range, returning a new AudioBuffer.
    pub fn slice(&self, start_frame: usize, end_frame: usize) -> Result<Self, AudioBufferError> {
        let ch = self.channels as usize;
        let start_sample = start_frame * ch;
        let end_sample = end_frame * ch;
        if end_sample > self.samples.len() || start_frame > end_frame {
            return Err(AudioBufferError::SliceOutOfBounds);
        }
        Ok(Self {
            samples: self.samples[start_sample..end_sample].to_vec(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        })
    }

    /// Convert stereo to mono by averaging channels. Returns self if already mono.
    pub fn to_mono(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }
        let ch = self.channels as usize;
        let mono: Vec<f32> = (0..self.num_frames())
            .map(|i| {
                let start = i * ch;
                let sum: f32 = self.samples[start..start + ch].iter().sum();
                sum / ch as f32
            })
            .collect();
        Self {
            samples: mono,
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Create an empty buffer with the given specs.
    pub fn empty(sample_rate: u32, channels: u16) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
            channels,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AudioBufferError {
    InvalidChannels,
    InvalidSampleRate,
    SampleCountMismatch { samples: usize, channels: u16 },
    SliceOutOfBounds,
}

impl std::fmt::Display for AudioBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidChannels => write!(f, "channels must be > 0"),
            Self::InvalidSampleRate => write!(f, "sample rate must be > 0"),
            Self::SampleCountMismatch { samples, channels } => {
                write!(f, "sample count {samples} not divisible by {channels} channels")
            }
            Self::SliceOutOfBounds => write!(f, "slice range out of bounds"),
        }
    }
}

impl std::error::Error for AudioBufferError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_mono_buffer() {
        let buf = AudioBuffer::new(vec![0.0, 0.5, 1.0, -1.0], 44100, 1).unwrap();
        assert_eq!(buf.num_frames(), 4);
        assert_eq!(buf.channels(), 1);
        assert_eq!(buf.sample_rate(), 44100);
    }

    #[test]
    fn create_stereo_buffer() {
        let buf = AudioBuffer::new(vec![0.0, 0.1, 0.5, 0.6], 44100, 2).unwrap();
        assert_eq!(buf.num_frames(), 2);
        assert_eq!(buf.channels(), 2);
    }

    #[test]
    fn reject_zero_channels() {
        let err = AudioBuffer::new(vec![1.0], 44100, 0).unwrap_err();
        assert_eq!(err, AudioBufferError::InvalidChannels);
    }

    #[test]
    fn reject_zero_sample_rate() {
        let err = AudioBuffer::new(vec![1.0], 0, 1).unwrap_err();
        assert_eq!(err, AudioBufferError::InvalidSampleRate);
    }

    #[test]
    fn reject_mismatched_sample_count() {
        let err = AudioBuffer::new(vec![0.0, 0.1, 0.2], 44100, 2).unwrap_err();
        assert!(matches!(err, AudioBufferError::SampleCountMismatch { .. }));
    }

    #[test]
    fn duration_calculation() {
        let buf = AudioBuffer::new(vec![0.0; 44100], 44100, 1).unwrap();
        assert!((buf.duration_secs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn duration_stereo() {
        // 44100 frames * 2 channels = 88200 samples = 1 second
        let buf = AudioBuffer::new(vec![0.0; 88200], 44100, 2).unwrap();
        assert!((buf.duration_secs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn get_frame_mono() {
        let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3], 44100, 1).unwrap();
        assert_eq!(buf.get_frame(0), Some(&[0.1f32][..]));
        assert_eq!(buf.get_frame(2), Some(&[0.3f32][..]));
        assert_eq!(buf.get_frame(3), None);
    }

    #[test]
    fn get_frame_stereo() {
        let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3, 0.4], 44100, 2).unwrap();
        assert_eq!(buf.get_frame(0), Some(&[0.1f32, 0.2][..]));
        assert_eq!(buf.get_frame(1), Some(&[0.3f32, 0.4][..]));
        assert_eq!(buf.get_frame(2), None);
    }

    #[test]
    fn slice_buffer() {
        let buf = AudioBuffer::new(vec![0.0, 1.0, 2.0, 3.0], 44100, 1).unwrap();
        let sliced = buf.slice(1, 3).unwrap();
        assert_eq!(sliced.samples(), &[1.0, 2.0]);
        assert_eq!(sliced.num_frames(), 2);
    }

    #[test]
    fn slice_out_of_bounds() {
        let buf = AudioBuffer::new(vec![0.0, 1.0], 44100, 1).unwrap();
        assert!(buf.slice(0, 5).is_err());
    }

    #[test]
    fn to_mono_from_stereo() {
        let buf = AudioBuffer::new(vec![0.0, 1.0, 0.4, 0.6], 44100, 2).unwrap();
        let mono = buf.to_mono();
        assert_eq!(mono.channels(), 1);
        assert_eq!(mono.num_frames(), 2);
        assert!((mono.samples()[0] - 0.5).abs() < 1e-6);
        assert!((mono.samples()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn to_mono_already_mono() {
        let buf = AudioBuffer::new(vec![0.5, 1.0], 44100, 1).unwrap();
        let mono = buf.to_mono();
        assert_eq!(mono.samples(), buf.samples());
    }

    #[test]
    fn empty_buffer() {
        let buf = AudioBuffer::empty(44100, 2);
        assert_eq!(buf.num_frames(), 0);
        assert!((buf.duration_secs() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn empty_samples_valid() {
        let buf = AudioBuffer::new(vec![], 44100, 2).unwrap();
        assert_eq!(buf.num_frames(), 0);
    }
}
