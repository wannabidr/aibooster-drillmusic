//! Tauri IPC commands for MIDI hardware communication.

use serde::{Deserialize, Serialize};

/// DTO for MIDI device information returned to the frontend.
#[derive(Serialize, Deserialize, Clone)]
pub struct MidiDeviceDto {
    pub port_id: String,
    pub name: String,
    pub device_type: String,
    pub has_output: bool,
}

/// DTO for MIDI configuration from the frontend.
#[derive(Serialize, Deserialize)]
pub struct MidiConfigDto {
    pub bpm_channel: u8,
    pub bpm_cc: u8,
    pub key_channel: u8,
    pub key_cc: u8,
    pub energy_channel: u8,
    pub energy_cc: u8,
    pub send_clock: bool,
}

/// DTO for track info to send to hardware.
#[derive(Serialize, Deserialize)]
pub struct MidiTrackInfoDto {
    pub bpm: f64,
    pub key: String,
    pub energy: u8,
    pub title: String,
    pub artist: String,
}

/// Result of a MIDI connection attempt.
#[derive(Serialize, Deserialize)]
pub struct MidiConnectionResult {
    pub connected: bool,
    pub device: Option<MidiDeviceDto>,
    pub protocol: String,
    pub error: Option<String>,
}

/// List available MIDI output devices.
///
/// Returns a list of detected MIDI devices with their types.
/// This scans all system MIDI output ports and identifies
/// known DJ hardware (Pioneer, Denon, NI) by name.
pub fn list_midi_devices() -> Result<Vec<MidiDeviceDto>, String> {
    // When ai-dj-dsp crate is linked:
    // let engine = ai_dj_dsp::midi::MidiEngine::new();
    // let devices = engine.list_devices().map_err(|e| e.to_string())?;
    // Ok(devices.into_iter().map(|d| MidiDeviceDto { ... }).collect())

    // Stub: return empty list until crate dependency is wired
    Ok(Vec::new())
}

/// Connect to a MIDI device by name.
///
/// Performs case-insensitive partial matching against available
/// MIDI port names. Auto-detects the device type and selects
/// the appropriate protocol handler (Pioneer/Denon/NI/Generic).
pub fn connect_midi_device(device_name: &str) -> Result<MidiConnectionResult, String> {
    // Stub until wired
    let _ = device_name;
    Ok(MidiConnectionResult {
        connected: false,
        device: None,
        protocol: "none".to_string(),
        error: Some("MIDI engine not yet initialized".to_string()),
    })
}

/// Disconnect from the current MIDI device.
pub fn disconnect_midi_device() -> Result<(), String> {
    // Stub
    Ok(())
}

/// Send track info (BPM, key, energy) to connected hardware.
///
/// Uses the auto-detected protocol handler to send the data
/// in the correct format for the connected hardware.
pub fn send_track_info_to_midi(info: MidiTrackInfoDto) -> Result<(), String> {
    let _ = info;
    Err("MIDI engine not yet initialized".to_string())
}

/// Update MIDI output configuration.
///
/// Allows the frontend to customize MIDI channel and CC assignments.
pub fn update_midi_config(config: MidiConfigDto) -> Result<(), String> {
    let _ = config;
    Ok(())
}
