//! Domain types for audio output routing.

use serde::{Deserialize, Serialize};

use crate::domain::AudioBuffer;

/// Represents an available audio output device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutputDevice {
    /// Unique device identifier.
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Number of output channels.
    pub channels: u16,
    /// Default sample rate in Hz.
    pub sample_rate: u32,
    /// Whether this is the system default output.
    pub is_default: bool,
}

/// Current playback state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioPlaybackState {
    /// No audio is playing.
    Stopped,
    /// Audio is currently playing.
    Playing,
    /// Playback is paused (can be resumed).
    Paused,
}

/// Port trait for audio output operations.
/// Domain layer -- no external dependencies.
///
/// Note: Not Send + Sync because platform audio streams (cpal::Stream)
/// are !Send !Sync. Wrap in Mutex at the Tauri command layer.
pub trait AudioOutputPort {
    /// List all available audio output devices.
    fn list_outputs(&self) -> Result<Vec<AudioOutputDevice>, AudioError>;

    /// Get the currently selected output device.
    fn current_output(&self) -> Option<&AudioOutputDevice>;

    /// Select an output device by ID.
    fn set_output(&mut self, device_id: &str) -> Result<(), AudioError>;

    /// Play an audio buffer through the selected output.
    fn play(&mut self, buffer: &AudioBuffer) -> Result<(), AudioError>;

    /// Stop playback.
    fn stop(&mut self) -> Result<(), AudioError>;

    /// Get current playback state.
    fn state(&self) -> AudioPlaybackState;
}

/// Errors from audio output operations.
#[derive(Debug, Clone)]
pub enum AudioError {
    /// No audio output devices found.
    NoDevices,
    /// Specified device not found.
    DeviceNotFound(String),
    /// Failed to open audio stream.
    StreamError(String),
    /// Failed to play audio.
    PlaybackError(String),
    /// Audio buffer format mismatch.
    FormatMismatch {
        expected_rate: u32,
        got_rate: u32,
    },
    /// Not currently playing.
    NotPlaying,
    /// No output device selected.
    NoOutputSelected,
    /// Platform-specific error.
    PlatformError(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevices => write!(f, "no audio output devices found"),
            Self::DeviceNotFound(id) => write!(f, "audio device not found: '{id}'"),
            Self::StreamError(msg) => write!(f, "audio stream error: {msg}"),
            Self::PlaybackError(msg) => write!(f, "playback error: {msg}"),
            Self::FormatMismatch {
                expected_rate,
                got_rate,
            } => write!(
                f,
                "sample rate mismatch: device expects {expected_rate} Hz, got {got_rate} Hz"
            ),
            Self::NotPlaying => write!(f, "not currently playing"),
            Self::NoOutputSelected => write!(f, "no output device selected"),
            Self::PlatformError(msg) => write!(f, "platform audio error: {msg}"),
        }
    }
}

impl std::error::Error for AudioError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_output_device_serialization() {
        let device = AudioOutputDevice {
            id: "dev-1".to_string(),
            name: "Built-in Output".to_string(),
            channels: 2,
            sample_rate: 44100,
            is_default: true,
        };
        let json = serde_json::to_string(&device).unwrap();
        let deserialized: AudioOutputDevice = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "dev-1");
        assert_eq!(deserialized.channels, 2);
        assert!(deserialized.is_default);
    }

    #[test]
    fn playback_state_values() {
        assert_ne!(AudioPlaybackState::Stopped, AudioPlaybackState::Playing);
        assert_ne!(AudioPlaybackState::Playing, AudioPlaybackState::Paused);
        assert_ne!(AudioPlaybackState::Paused, AudioPlaybackState::Stopped);
    }

    #[test]
    fn error_display_messages() {
        assert_eq!(
            AudioError::NoDevices.to_string(),
            "no audio output devices found"
        );
        assert_eq!(
            AudioError::DeviceNotFound("x".to_string()).to_string(),
            "audio device not found: 'x'"
        );
        assert_eq!(
            AudioError::FormatMismatch {
                expected_rate: 48000,
                got_rate: 44100
            }
            .to_string(),
            "sample rate mismatch: device expects 48000 Hz, got 44100 Hz"
        );
        assert_eq!(AudioError::NotPlaying.to_string(), "not currently playing");
        assert_eq!(
            AudioError::NoOutputSelected.to_string(),
            "no output device selected"
        );
    }
}
