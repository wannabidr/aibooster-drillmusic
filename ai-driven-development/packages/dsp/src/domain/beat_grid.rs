/// Represents detected beat positions aligned to a tempo.
#[derive(Debug, Clone)]
pub struct BeatGrid {
    bpm: f64,
    beat_positions: Vec<u64>,
    downbeat_indices: Vec<usize>,
    time_signature: (u8, u8),
}

impl BeatGrid {
    pub fn new(
        bpm: f64,
        beat_positions: Vec<u64>,
        downbeat_indices: Vec<usize>,
        time_signature: (u8, u8),
    ) -> Result<Self, BeatGridError> {
        if bpm <= 0.0 || !bpm.is_finite() {
            return Err(BeatGridError::InvalidBpm(bpm));
        }
        if time_signature.0 == 0 || time_signature.1 == 0 {
            return Err(BeatGridError::InvalidTimeSignature(time_signature));
        }
        for &idx in &downbeat_indices {
            if idx >= beat_positions.len() && !beat_positions.is_empty() {
                return Err(BeatGridError::DownbeatOutOfRange {
                    index: idx,
                    beat_count: beat_positions.len(),
                });
            }
        }
        // Verify beat positions are sorted
        for w in beat_positions.windows(2) {
            if w[0] >= w[1] {
                return Err(BeatGridError::UnsortedBeats);
            }
        }
        Ok(Self {
            bpm,
            beat_positions,
            downbeat_indices,
            time_signature,
        })
    }

    pub fn bpm(&self) -> f64 {
        self.bpm
    }

    pub fn beat_positions(&self) -> &[u64] {
        &self.beat_positions
    }

    pub fn downbeat_indices(&self) -> &[usize] {
        &self.downbeat_indices
    }

    pub fn time_signature(&self) -> (u8, u8) {
        self.time_signature
    }

    pub fn beat_count(&self) -> usize {
        self.beat_positions.len()
    }

    pub fn beats_per_bar(&self) -> u8 {
        self.time_signature.0
    }

    pub fn bar_count(&self) -> usize {
        let bpb = self.beats_per_bar() as usize;
        if bpb == 0 {
            return 0;
        }
        self.beat_positions.len() / bpb
    }

    /// Find the index of the nearest beat to a given sample position.
    pub fn nearest_beat(&self, sample_pos: u64) -> Option<usize> {
        if self.beat_positions.is_empty() {
            return None;
        }
        let idx = self.beat_positions.partition_point(|&b| b < sample_pos);
        if idx == 0 {
            return Some(0);
        }
        if idx >= self.beat_positions.len() {
            return Some(self.beat_positions.len() - 1);
        }
        let dist_before = sample_pos - self.beat_positions[idx - 1];
        let dist_after = self.beat_positions[idx] - sample_pos;
        if dist_before <= dist_after {
            Some(idx - 1)
        } else {
            Some(idx)
        }
    }

    /// Find the index of the nearest downbeat to a given sample position.
    pub fn nearest_downbeat(&self, sample_pos: u64) -> Option<usize> {
        if self.downbeat_indices.is_empty() {
            return None;
        }
        let mut best = self.downbeat_indices[0];
        let mut best_dist = u64::MAX;
        for &di in &self.downbeat_indices {
            if di >= self.beat_positions.len() {
                continue;
            }
            let beat_pos = self.beat_positions[di];
            let dist = beat_pos.abs_diff(sample_pos);
            if dist < best_dist {
                best_dist = dist;
                best = di;
            }
        }
        Some(best)
    }

    /// Generate a beat grid from BPM and a known first beat position.
    pub fn from_bpm(bpm: f64, first_beat_sample: u64, sample_rate: u32, total_frames: u64) -> Result<Self, BeatGridError> {
        if bpm <= 0.0 || !bpm.is_finite() {
            return Err(BeatGridError::InvalidBpm(bpm));
        }
        let samples_per_beat = (sample_rate as f64 * 60.0 / bpm) as u64;
        if samples_per_beat == 0 {
            return Err(BeatGridError::InvalidBpm(bpm));
        }

        let mut beats = Vec::new();
        let mut pos = first_beat_sample;
        while pos < total_frames {
            beats.push(pos);
            pos += samples_per_beat;
        }

        let downbeats: Vec<usize> = (0..beats.len()).step_by(4).collect();

        Self::new(bpm, beats, downbeats, (4, 4))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BeatGridError {
    InvalidBpm(f64),
    InvalidTimeSignature((u8, u8)),
    DownbeatOutOfRange { index: usize, beat_count: usize },
    UnsortedBeats,
}

impl std::fmt::Display for BeatGridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBpm(bpm) => write!(f, "invalid BPM: {bpm}"),
            Self::InvalidTimeSignature(ts) => write!(f, "invalid time signature: {}/{}", ts.0, ts.1),
            Self::DownbeatOutOfRange { index, beat_count } => {
                write!(f, "downbeat index {index} out of range (beat count: {beat_count})")
            }
            Self::UnsortedBeats => write!(f, "beat positions must be sorted in ascending order"),
        }
    }
}

impl std::error::Error for BeatGridError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_valid_beat_grid() {
        let grid = BeatGrid::new(128.0, vec![0, 100, 200, 300], vec![0], (4, 4)).unwrap();
        assert_eq!(grid.bpm(), 128.0);
        assert_eq!(grid.beat_count(), 4);
        assert_eq!(grid.beats_per_bar(), 4);
        assert_eq!(grid.bar_count(), 1);
    }

    #[test]
    fn reject_zero_bpm() {
        assert!(BeatGrid::new(0.0, vec![], vec![], (4, 4)).is_err());
    }

    #[test]
    fn reject_negative_bpm() {
        assert!(BeatGrid::new(-120.0, vec![], vec![], (4, 4)).is_err());
    }

    #[test]
    fn reject_nan_bpm() {
        assert!(BeatGrid::new(f64::NAN, vec![], vec![], (4, 4)).is_err());
    }

    #[test]
    fn reject_invalid_time_signature() {
        assert!(BeatGrid::new(120.0, vec![], vec![], (0, 4)).is_err());
        assert!(BeatGrid::new(120.0, vec![], vec![], (4, 0)).is_err());
    }

    #[test]
    fn reject_unsorted_beats() {
        assert!(BeatGrid::new(120.0, vec![200, 100, 300], vec![], (4, 4)).is_err());
    }

    #[test]
    fn reject_duplicate_beat_positions() {
        assert!(BeatGrid::new(120.0, vec![100, 100, 200], vec![], (4, 4)).is_err());
    }

    #[test]
    fn reject_downbeat_out_of_range() {
        assert!(BeatGrid::new(120.0, vec![0, 100], vec![5], (4, 4)).is_err());
    }

    #[test]
    fn nearest_beat_exact() {
        let grid = BeatGrid::new(120.0, vec![0, 1000, 2000, 3000], vec![0], (4, 4)).unwrap();
        assert_eq!(grid.nearest_beat(1000), Some(1));
    }

    #[test]
    fn nearest_beat_between() {
        let grid = BeatGrid::new(120.0, vec![0, 1000, 2000], vec![0], (4, 4)).unwrap();
        // 600 is closer to 1000 than 0
        assert_eq!(grid.nearest_beat(600), Some(1));
        // 400 is closer to 0 than 1000
        assert_eq!(grid.nearest_beat(400), Some(0));
    }

    #[test]
    fn nearest_beat_before_first() {
        let grid = BeatGrid::new(120.0, vec![1000, 2000], vec![0], (4, 4)).unwrap();
        assert_eq!(grid.nearest_beat(0), Some(0));
    }

    #[test]
    fn nearest_beat_after_last() {
        let grid = BeatGrid::new(120.0, vec![0, 1000], vec![0], (4, 4)).unwrap();
        assert_eq!(grid.nearest_beat(5000), Some(1));
    }

    #[test]
    fn nearest_beat_empty() {
        let grid = BeatGrid::new(120.0, vec![], vec![], (4, 4)).unwrap();
        assert_eq!(grid.nearest_beat(500), None);
    }

    #[test]
    fn nearest_downbeat() {
        let grid = BeatGrid::new(
            120.0,
            vec![0, 1000, 2000, 3000, 4000, 5000, 6000, 7000],
            vec![0, 4],
            (4, 4),
        )
        .unwrap();
        // sample 3500 is closest to downbeat at index 4 (position 4000)
        assert_eq!(grid.nearest_downbeat(3500), Some(4));
        // sample 500 is closest to downbeat at index 0 (position 0)
        assert_eq!(grid.nearest_downbeat(500), Some(0));
    }

    #[test]
    fn from_bpm_generates_grid() {
        // 120 BPM at 44100 Hz = 22050 samples per beat
        let grid = BeatGrid::from_bpm(120.0, 0, 44100, 44100 * 4).unwrap();
        assert_eq!(grid.bpm(), 120.0);
        // 4 seconds at 120 BPM = 8 beats
        assert_eq!(grid.beat_count(), 8);
        assert_eq!(grid.beat_positions()[0], 0);
        // Downbeats at every 4th beat
        assert_eq!(grid.downbeat_indices(), &[0, 4]);
    }

    #[test]
    fn from_bpm_128() {
        // 128 BPM at 44100 Hz = 20671.875 -> 20671 samples per beat
        let grid = BeatGrid::from_bpm(128.0, 0, 44100, 44100).unwrap();
        assert!(grid.beat_count() > 0);
        let expected_spb = (44100.0 * 60.0 / 128.0) as u64;
        assert_eq!(grid.beat_positions()[1], expected_spb);
    }

    #[test]
    fn bar_count_calculation() {
        let grid = BeatGrid::new(
            120.0,
            vec![0, 100, 200, 300, 400, 500, 600, 700, 800],
            vec![0, 4, 8],
            (4, 4),
        )
        .unwrap();
        // 9 beats / 4 beats per bar = 2 full bars
        assert_eq!(grid.bar_count(), 2);
    }
}
