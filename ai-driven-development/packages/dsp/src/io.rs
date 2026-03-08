use crate::domain::{AudioBuffer, AudioBufferError};
use std::path::Path;

/// Errors that can occur during audio file I/O.
#[derive(Debug)]
pub enum IoError {
    Hound(hound::Error),
    Buffer(AudioBufferError),
    UnsupportedFormat(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hound(e) => write!(f, "WAV I/O error: {e}"),
            Self::Buffer(e) => write!(f, "buffer error: {e}"),
            Self::UnsupportedFormat(msg) => write!(f, "unsupported format: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<hound::Error> for IoError {
    fn from(e: hound::Error) -> Self {
        IoError::Hound(e)
    }
}

impl From<AudioBufferError> for IoError {
    fn from(e: AudioBufferError) -> Self {
        IoError::Buffer(e)
    }
}

/// Load a WAV file into an AudioBuffer.
/// Supports 16-bit int, 24-bit int, 32-bit int, and 32-bit float formats.
/// All formats are normalized to f32 in the range [-1.0, 1.0].
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer, IoError> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels;
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            reader
                .into_samples::<i16>()
                .map(|s| s.map(|v| v as f32 / i16::MAX as f32))
                .collect::<Result<_, _>>()?
        }
        (hound::SampleFormat::Int, 24) => {
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / 8_388_607.0)) // 2^23 - 1
                .collect::<Result<_, _>>()?
        }
        (hound::SampleFormat::Int, 32) => {
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / i32::MAX as f32))
                .collect::<Result<_, _>>()?
        }
        (hound::SampleFormat::Float, 32) => {
            reader
                .into_samples::<f32>()
                .collect::<Result<_, _>>()?
        }
        (fmt, bits) => {
            return Err(IoError::UnsupportedFormat(format!(
                "{fmt:?} {bits}-bit not supported"
            )));
        }
    };

    Ok(AudioBuffer::new(samples, sample_rate, channels)?)
}

/// Save an AudioBuffer to a WAV file (16-bit PCM, 44.1kHz).
pub fn save_wav<P: AsRef<Path>>(buffer: &AudioBuffer, path: P) -> Result<(), IoError> {
    let spec = hound::WavSpec {
        channels: buffer.channels(),
        sample_rate: buffer.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in buffer.samples() {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * i16::MAX as f32) as i16;
        writer.write_sample(int_sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("ai_dj_dsp_tests");
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn roundtrip_mono_wav() {
        let path = temp_dir().join("test_mono.wav");
        let original = AudioBuffer::new(
            vec![0.0, 0.5, -0.5, 1.0, -1.0],
            44100,
            1,
        ).unwrap();

        save_wav(&original, &path).unwrap();
        let loaded = load_wav(&path).unwrap();

        assert_eq!(loaded.sample_rate(), 44100);
        assert_eq!(loaded.channels(), 1);
        assert_eq!(loaded.num_frames(), 5);

        // 16-bit quantization means we lose some precision
        for (orig, loaded) in original.samples().iter().zip(loaded.samples().iter()) {
            assert!(
                (orig - loaded).abs() < 0.001,
                "sample mismatch: orig={orig}, loaded={loaded}"
            );
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_stereo_wav() {
        let path = temp_dir().join("test_stereo.wav");
        let original = AudioBuffer::new(
            vec![0.1, -0.1, 0.5, -0.5, 0.9, -0.9, 0.0, 0.0],
            44100,
            2,
        ).unwrap();

        save_wav(&original, &path).unwrap();
        let loaded = load_wav(&path).unwrap();

        assert_eq!(loaded.channels(), 2);
        assert_eq!(loaded.num_frames(), 4);

        for (orig, loaded) in original.samples().iter().zip(loaded.samples().iter()) {
            assert!(
                (orig - loaded).abs() < 0.001,
                "sample mismatch: orig={orig}, loaded={loaded}"
            );
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn save_clamps_out_of_range_samples() {
        let path = temp_dir().join("test_clamp.wav");
        let buf = AudioBuffer::new(vec![2.0, -2.0, 0.5], 44100, 1).unwrap();

        save_wav(&buf, &path).unwrap();
        let loaded = load_wav(&path).unwrap();

        // Out-of-range samples should be clamped to [-1.0, 1.0]
        assert!((loaded.samples()[0] - 1.0).abs() < 0.001);
        assert!((loaded.samples()[1] - (-1.0)).abs() < 0.001);
        assert!((loaded.samples()[2] - 0.5).abs() < 0.001);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn load_nonexistent_file_returns_error() {
        let result = load_wav("/nonexistent/path/audio.wav");
        assert!(result.is_err());
    }

    #[test]
    fn save_and_load_preserves_sample_rate() {
        let path = temp_dir().join("test_48k.wav");
        let buf = AudioBuffer::new(vec![0.0; 48000], 48000, 1).unwrap();

        save_wav(&buf, &path).unwrap();
        let loaded = load_wav(&path).unwrap();

        assert_eq!(loaded.sample_rate(), 48000);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn roundtrip_silence() {
        let path = temp_dir().join("test_silence.wav");
        let buf = AudioBuffer::new(vec![0.0; 44100], 44100, 1).unwrap();

        save_wav(&buf, &path).unwrap();
        let loaded = load_wav(&path).unwrap();

        assert_eq!(loaded.num_frames(), 44100);
        for s in loaded.samples() {
            assert!((s - 0.0).abs() < 0.001);
        }

        fs::remove_file(&path).ok();
    }
}
