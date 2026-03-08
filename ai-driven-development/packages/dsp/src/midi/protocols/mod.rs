//! Protocol handlers for specific DJ hardware.

pub mod pioneer;
pub mod denon;
pub mod native_instruments;

use super::types::{MidiConfig, MidiError, MidiMessage, TrackInfo};

/// Trait for hardware-specific protocol handlers.
/// Each handler knows how to translate track metadata into
/// the correct MIDI messages for its hardware.
pub trait ProtocolHandler: Send + Sync {
    /// Get the protocol display name.
    fn name(&self) -> &str;

    /// Generate MIDI messages to send BPM to hardware.
    fn send_bpm(&self, bpm: f64, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError>;

    /// Generate MIDI messages to send key information to hardware.
    fn send_key(&self, key: &str, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError>;

    /// Generate MIDI messages to send energy level to hardware.
    fn send_energy(&self, energy: u8, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError>;

    /// Generate all messages needed to send full track info.
    fn send_track_info(
        &self,
        info: &TrackInfo,
        config: &MidiConfig,
    ) -> Result<Vec<MidiMessage>, MidiError> {
        let mut messages = Vec::new();
        messages.extend(self.send_bpm(info.bpm, config)?);
        messages.extend(self.send_key(&info.key, config)?);
        messages.extend(self.send_energy(info.energy, config)?);
        Ok(messages)
    }

    /// Generate MIDI clock messages for a given BPM.
    /// Returns the interval in microseconds between clock ticks.
    fn clock_interval_us(&self, bpm: f64) -> u64 {
        // MIDI clock: 24 ppqn (pulses per quarter note)
        let beats_per_second = bpm / 60.0;
        let pulses_per_second = beats_per_second * 24.0;
        (1_000_000.0 / pulses_per_second) as u64
    }
}

/// Map a musical key string to a MIDI note number.
/// Uses Camelot wheel mapping: 1A-12A, 1B-12B mapped to notes 0-23.
pub fn key_to_midi_note(key: &str) -> u8 {
    match key.to_uppercase().as_str() {
        // Minor keys (A series in Camelot)
        "ABM" | "G#M" | "1A" => 0,
        "EBM" | "D#M" | "2A" => 1,
        "BBM" | "A#M" | "3A" => 2,
        "FM" | "4A" => 3,
        "CM" | "5A" => 4,
        "GM" | "6A" => 5,
        "DM" | "7A" => 6,
        "AM" | "8A" => 7,
        "EM" | "9A" => 8,
        "BM" | "10A" => 9,
        "F#M" | "GBM" | "11A" => 10,
        "DBM" | "C#M" | "12A" => 11,
        // Major keys (B series in Camelot)
        "B" | "1B" => 12,
        "F#" | "GB" | "2B" => 13,
        "DB" | "C#" | "3B" => 14,
        "AB" | "G#" | "4B" => 15,
        "EB" | "D#" | "5B" => 16,
        "BB" | "A#" | "6B" => 17,
        "F" | "7B" => 18,
        "C" | "8B" => 19,
        "G" | "9B" => 20,
        "D" | "10B" => 21,
        "A" | "11B" => 22,
        "E" | "12B" => 23,
        _ => 0, // Unknown key defaults to 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_to_midi_note_minor_keys() {
        assert_eq!(key_to_midi_note("Am"), 7);  // 8A
        assert_eq!(key_to_midi_note("Cm"), 4);  // 5A
        assert_eq!(key_to_midi_note("Fm"), 3);  // 4A
    }

    #[test]
    fn key_to_midi_note_major_keys() {
        assert_eq!(key_to_midi_note("C"), 19);  // 8B
        assert_eq!(key_to_midi_note("G"), 20);  // 9B
        assert_eq!(key_to_midi_note("F"), 18);  // 7B
    }

    #[test]
    fn key_to_midi_note_camelot() {
        assert_eq!(key_to_midi_note("8A"), 7);
        assert_eq!(key_to_midi_note("8B"), 19);
        assert_eq!(key_to_midi_note("1A"), 0);
        assert_eq!(key_to_midi_note("12B"), 23);
    }

    #[test]
    fn key_to_midi_note_unknown() {
        assert_eq!(key_to_midi_note("??"), 0);
    }
}
