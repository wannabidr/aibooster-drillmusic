//! Virtual audio cable routing for preview playback.
//!
//! Routes AI blend preview audio to a secondary output device (e.g., DJ headphone
//! channel) without disrupting the main audio output. Uses the `cpal` crate for
//! cross-platform audio output (CoreAudio on macOS, WASAPI on Windows).

pub mod cpal_output;
pub mod types;

pub use cpal_output::CpalAudioOutput;
pub use types::{AudioError, AudioOutputDevice, AudioOutputPort, AudioPlaybackState};
