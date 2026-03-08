pub mod analysis;
pub mod audio;
pub mod midi;

// Re-export analysis types for backward compatibility
pub use analysis::{
    analyze_track, batch_analyze, get_recommendations, import_library, AnalysisResultDto,
    ImportResult, RecommendationDto, TrackDto,
};
