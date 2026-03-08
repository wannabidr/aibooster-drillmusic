//! Tauri IPC commands for virtual audio cable routing.

use serde::{Deserialize, Serialize};

/// DTO for audio output device returned to the frontend.
#[derive(Serialize, Deserialize, Clone)]
pub struct AudioDeviceDto {
    pub id: String,
    pub name: String,
    pub channels: u16,
    pub sample_rate: u32,
    pub is_default: bool,
}

/// DTO for audio playback state.
#[derive(Serialize, Deserialize)]
pub struct AudioStateDto {
    pub state: String,
    pub device: Option<AudioDeviceDto>,
}

/// List available audio output devices.
///
/// Returns all system audio output devices with their capabilities.
/// Used by the frontend to populate the audio device selection UI.
pub fn list_audio_devices() -> Result<Vec<AudioDeviceDto>, String> {
    // When ai-dj-dsp crate is linked:
    // let output = ai_dj_dsp::audio_output::CpalAudioOutput::new();
    // let devices = output.list_outputs().map_err(|e| e.to_string())?;
    // Ok(devices.into_iter().map(|d| AudioDeviceDto { ... }).collect())

    // Stub until wired
    Ok(Vec::new())
}

/// Set the preview output device.
///
/// Routes AI blend preview audio to the specified device,
/// allowing DJs to preview in headphones while the main
/// output remains undisturbed.
pub fn set_preview_output(device_id: &str) -> Result<(), String> {
    let _ = device_id;
    Err("Audio output not yet initialized".to_string())
}

/// Play an audio preview through the selected output.
pub fn play_preview(file_path: &str) -> Result<(), String> {
    let _ = file_path;
    Err("Audio output not yet initialized".to_string())
}

/// Stop preview playback.
pub fn stop_preview() -> Result<(), String> {
    Err("Audio output not yet initialized".to_string())
}

/// Get current audio playback state.
pub fn get_audio_state() -> Result<AudioStateDto, String> {
    Ok(AudioStateDto {
        state: "stopped".to_string(),
        device: None,
    })
}
