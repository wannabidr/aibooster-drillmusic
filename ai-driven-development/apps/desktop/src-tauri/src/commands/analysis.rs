use serde::{Deserialize, Serialize};

use crate::sidecar::SidecarManager;

// --- DTOs ---

#[derive(Serialize, Deserialize)]
pub struct TrackDto {
    pub id: String,
    pub title: String,
    pub artist: String,
    pub album: Option<String>,
    pub file_path: String,
    pub duration_ms: Option<u64>,
    pub bpm: Option<f64>,
    pub key: Option<String>,
    pub camelot_position: Option<String>,
    pub energy: Option<u8>,
    pub genre: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct AnalysisResultDto {
    pub track_id: String,
    pub bpm: f64,
    pub key: String,
    pub camelot_position: String,
    pub energy: u8,
    pub fingerprint: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct RecommendationDto {
    pub track_id: String,
    pub score: f64,
    pub bpm_score: f64,
    pub key_score: f64,
    pub energy_score: f64,
    pub genre_score: f64,
    pub history_score: f64,
    pub confidence: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ImportResult {
    pub track_count: usize,
    pub errors: Vec<String>,
}

// --- Tauri IPC Commands ---

/// Import a library from DJ software or folder.
pub fn import_library(
    sidecar: &SidecarManager,
    source: &str,
    path: &str,
) -> Result<ImportResult, String> {
    let params = serde_json::json!({
        "source": source,
        "path": path,
    });
    let result = sidecar.call("import_library", params)?;
    serde_json::from_value(result).map_err(|e| format!("Deserialize error: {}", e))
}

/// Analyze a single track.
pub fn analyze_track(
    sidecar: &SidecarManager,
    track_id: &str,
    file_path: &str,
) -> Result<AnalysisResultDto, String> {
    let params = serde_json::json!({
        "track_id": track_id,
        "file_path": file_path,
    });
    let result = sidecar.call("analyze_track", params)?;
    serde_json::from_value(result).map_err(|e| format!("Deserialize error: {}", e))
}

/// Batch analyze multiple tracks.
pub fn batch_analyze(
    sidecar: &SidecarManager,
    track_ids: &[String],
) -> Result<serde_json::Value, String> {
    let params = serde_json::json!({
        "track_ids": track_ids,
    });
    sidecar.call("batch_analyze", params)
}

/// Get recommendations for the current track.
pub fn get_recommendations(
    sidecar: &SidecarManager,
    current_track_id: &str,
    limit: usize,
    exclude_track_ids: &[String],
) -> Result<Vec<RecommendationDto>, String> {
    let params = serde_json::json!({
        "current_track_id": current_track_id,
        "limit": limit,
        "exclude_track_ids": exclude_track_ids,
    });
    let result = sidecar.call("get_recommendations", params)?;
    serde_json::from_value(result).map_err(|e| format!("Deserialize error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_dto_serialization_roundtrip() {
        let track = TrackDto {
            id: "track-1".to_string(),
            title: "Test Track".to_string(),
            artist: "Test Artist".to_string(),
            album: Some("Test Album".to_string()),
            file_path: "/music/test.wav".to_string(),
            duration_ms: Some(240000),
            bpm: Some(128.0),
            key: Some("Am".to_string()),
            camelot_position: Some("8A".to_string()),
            energy: Some(7),
            genre: Some("house".to_string()),
        };

        let json = serde_json::to_string(&track).unwrap();
        let deserialized: TrackDto = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "track-1");
        assert_eq!(deserialized.bpm, Some(128.0));
        assert_eq!(deserialized.genre, Some("house".to_string()));
    }

    #[test]
    fn track_dto_optional_fields_null() {
        let json = r#"{"id":"t1","title":"T","artist":"A","album":null,"file_path":"/f","duration_ms":null,"bpm":null,"key":null,"camelot_position":null,"energy":null,"genre":null}"#;
        let track: TrackDto = serde_json::from_str(json).unwrap();
        assert!(track.album.is_none());
        assert!(track.bpm.is_none());
        assert!(track.energy.is_none());
    }

    #[test]
    fn analysis_result_dto_serialization() {
        let result = AnalysisResultDto {
            track_id: "track-1".to_string(),
            bpm: 126.5,
            key: "Cm".to_string(),
            camelot_position: "5A".to_string(),
            energy: 8,
            fingerprint: Some("AQAA...".to_string()),
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("126.5"));
        assert!(json.contains("\"5A\""));
    }

    #[test]
    fn recommendation_dto_serialization() {
        let rec = RecommendationDto {
            track_id: "rec-1".to_string(),
            score: 0.85,
            bpm_score: 0.9,
            key_score: 1.0,
            energy_score: 0.7,
            genre_score: 0.8,
            history_score: 0.6,
            confidence: 0.92,
        };

        let json = serde_json::to_string(&rec).unwrap();
        let deserialized: RecommendationDto = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.score, 0.85);
        assert_eq!(deserialized.confidence, 0.92);
    }

    #[test]
    fn import_result_serialization() {
        let result = ImportResult {
            track_count: 42,
            errors: vec!["missing file: track.mp3".to_string()],
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ImportResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.track_count, 42);
        assert_eq!(deserialized.errors.len(), 1);
    }

    #[test]
    fn import_result_empty_errors() {
        let result = ImportResult {
            track_count: 100,
            errors: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"errors\":[]"));
    }

    #[test]
    fn commands_fail_without_running_sidecar() {
        let mgr = SidecarManager::new();

        assert!(import_library(&mgr, "rekordbox", "/path").is_err());
        assert!(analyze_track(&mgr, "t1", "/path/track.wav").is_err());
        assert!(batch_analyze(&mgr, &["t1".to_string()]).is_err());
        assert!(get_recommendations(&mgr, "t1", 5, &[]).is_err());
    }

    #[test]
    fn track_dto_with_windows_path() {
        let track = TrackDto {
            id: "win-track".to_string(),
            title: "Windows Track".to_string(),
            artist: "Artist".to_string(),
            album: None,
            file_path: r"C:\Users\DJ\Music\track.wav".to_string(),
            duration_ms: Some(180000),
            bpm: Some(140.0),
            key: None,
            camelot_position: None,
            energy: None,
            genre: None,
        };

        let json = serde_json::to_string(&track).unwrap();
        let deserialized: TrackDto = serde_json::from_str(&json).unwrap();
        assert!(deserialized.file_path.contains("Music"));
    }

    #[test]
    fn track_dto_with_unicode_fields() {
        let track = TrackDto {
            id: "unicode-1".to_string(),
            title: "Nuit Blanche".to_string(),
            artist: "DJ 日本語".to_string(),
            album: Some("Album".to_string()),
            file_path: "/music/nuit.wav".to_string(),
            duration_ms: None,
            bpm: None,
            key: None,
            camelot_position: None,
            energy: None,
            genre: None,
        };

        let json = serde_json::to_string(&track).unwrap();
        let deserialized: TrackDto = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.artist, "DJ 日本語");
    }
}
