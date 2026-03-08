// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod sidecar;

use sidecar::SidecarManager;
use tauri::Manager;

#[tauri::command]
fn import_library(
    state: tauri::State<'_, SidecarManager>,
    source: String,
    path: String,
) -> Result<commands::ImportResult, String> {
    commands::import_library(&state, &source, &path)
}

#[tauri::command]
fn analyze_track(
    state: tauri::State<'_, SidecarManager>,
    track_id: String,
    file_path: String,
) -> Result<commands::AnalysisResultDto, String> {
    commands::analyze_track(&state, &track_id, &file_path)
}

#[tauri::command]
fn batch_analyze(
    state: tauri::State<'_, SidecarManager>,
    track_ids: Vec<String>,
) -> Result<serde_json::Value, String> {
    commands::batch_analyze(&state, &track_ids)
}

#[tauri::command]
fn get_recommendations(
    state: tauri::State<'_, SidecarManager>,
    current_track_id: String,
    limit: usize,
    exclude_track_ids: Vec<String>,
) -> Result<Vec<commands::RecommendationDto>, String> {
    commands::get_recommendations(&state, &current_track_id, limit, &exclude_track_ids)
}

#[tauri::command]
fn render_preview(
    track_a_path: String,
    track_b_path: String,
    bpm_a: f64,
    bpm_b: f64,
    transition_beats: u32,
    curve: String,
) -> Result<String, String> {
    ai_dj_dsp::ffi::render_crossfade(
        &track_a_path,
        &track_b_path,
        bpm_a,
        bpm_b,
        transition_beats,
        &curve,
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn render_ai_blend(
    track_a_path: String,
    track_b_path: String,
    bpm_a: f64,
    bpm_b: f64,
    blend_style: String,
    genre: String,
) -> Result<String, String> {
    ai_dj_dsp::ffi::render_ai_blend(
        &track_a_path,
        &track_b_path,
        bpm_a,
        bpm_b,
        &blend_style,
        &genre,
    )
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn health_check(state: tauri::State<'_, SidecarManager>) -> bool {
    state.health_check()
}

// --- MIDI Commands ---

#[tauri::command]
fn list_midi_devices() -> Result<Vec<commands::midi::MidiDeviceDto>, String> {
    commands::midi::list_midi_devices()
}

#[tauri::command]
fn connect_midi_device(device_name: String) -> Result<commands::midi::MidiConnectionResult, String> {
    commands::midi::connect_midi_device(&device_name)
}

#[tauri::command]
fn disconnect_midi_device() -> Result<(), String> {
    commands::midi::disconnect_midi_device()
}

#[tauri::command]
fn send_track_info_to_midi(info: commands::midi::MidiTrackInfoDto) -> Result<(), String> {
    commands::midi::send_track_info_to_midi(info)
}

fn main() {
    let sidecar = SidecarManager::new();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(sidecar)
        .setup(|app| {
            let sidecar_state = app.state::<SidecarManager>();

            // Resolve sidecar binary path (platform-aware)
            let sidecar_name = if cfg!(target_os = "windows") {
                "sidecars/ai-dj-analysis.exe"
            } else {
                "sidecars/ai-dj-analysis"
            };

            let resource_path = app
                .path()
                .resource_dir()
                .map(|p| p.join(sidecar_name))
                .ok();

            if let Some(path) = resource_path {
                if let Some(path_str) = path.to_str() {
                    if let Err(e) = sidecar_state.start(path_str) {
                        eprintln!("Warning: Failed to start sidecar: {}", e);
                        eprintln!("The app will run without analysis capabilities.");
                    }
                }
            } else {
                eprintln!("Warning: Could not resolve sidecar path.");
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            import_library,
            analyze_track,
            batch_analyze,
            get_recommendations,
            render_preview,
            render_ai_blend,
            health_check,
            list_midi_devices,
            connect_midi_device,
            disconnect_midi_device,
            send_track_info_to_midi,
        ])
        .run(tauri::generate_context!())
        .expect("error running tauri application");
}
