//! Denon SC6000 / SC5000 protocol handler.
//!
//! Denon DJ Engine OS hardware uses standard MIDI CC messages
//! for parameter feedback. The SC6000 supports both MIDI and
//! StagelinQ protocol over Ethernet.
//!
//! Denon SysEx Manufacturer ID: 0x00 0x02 0x0B (InMusic / Denon DJ)

use super::{key_to_midi_note, ProtocolHandler};
use crate::midi::types::{
    ControlNumber, MidiConfig, MidiError, MidiMessage, MidiValue, TrackInfo,
};

/// Denon DJ SysEx manufacturer ID (InMusic Brands).
const DENON_SYSEX_ID: [u8; 3] = [0x00, 0x02, 0x0B];

/// Denon-specific CC assignments.
const DENON_BPM_CC_MSB: u8 = 50; // BPM coarse
const DENON_BPM_CC_LSB: u8 = 51; // BPM fine
const DENON_KEY_CC: u8 = 52; // Musical key
const DENON_DECK_SELECT_CC: u8 = 53; // Active deck (0-3)

/// Protocol handler for Denon SC6000/SC5000 hardware.
pub struct DenonProtocol {
    /// Active deck number (0-3).
    active_deck: u8,
}

impl DenonProtocol {
    pub fn new() -> Self {
        Self { active_deck: 0 }
    }

    pub fn with_deck(deck: u8) -> Self {
        Self {
            active_deck: deck.min(3),
        }
    }

    /// Encode BPM as two 7-bit MIDI values (MSB/LSB).
    fn encode_bpm(bpm: f64) -> (u8, u8) {
        let bpm_int = (bpm * 100.0).round() as u16;
        let msb = ((bpm_int >> 7) & 0x7F) as u8;
        let lsb = (bpm_int & 0x7F) as u8;
        (msb, lsb)
    }

    /// Build a deck select message.
    fn deck_select_message(&self, config: &MidiConfig) -> Result<MidiMessage, MidiError> {
        Ok(MidiMessage::ControlChange(
            config.bpm_channel,
            ControlNumber::new(DENON_DECK_SELECT_CC)?,
            MidiValue::new(self.active_deck)?,
        ))
    }

    /// Build a Denon SysEx message for track metadata.
    fn build_sysex_track_info(&self, info: &TrackInfo) -> MidiMessage {
        let mut data = Vec::new();
        data.extend_from_slice(&DENON_SYSEX_ID);
        // Command: track update (0x20)
        data.push(0x20);
        // Deck number
        data.push(self.active_deck);
        // BPM
        let (msb, lsb) = Self::encode_bpm(info.bpm);
        data.push(msb);
        data.push(lsb);
        // Key
        data.push(key_to_midi_note(&info.key));
        // Energy
        data.push(info.energy.min(127));
        MidiMessage::SysEx(data)
    }
}

impl ProtocolHandler for DenonProtocol {
    fn name(&self) -> &str {
        "Denon SC6000"
    }

    fn send_bpm(&self, bpm: f64, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        let (msb, lsb) = Self::encode_bpm(bpm);
        Ok(vec![
            self.deck_select_message(config)?,
            MidiMessage::ControlChange(
                config.bpm_channel,
                ControlNumber::new(DENON_BPM_CC_MSB)?,
                MidiValue::new(msb)?,
            ),
            MidiMessage::ControlChange(
                config.bpm_channel,
                ControlNumber::new(DENON_BPM_CC_LSB)?,
                MidiValue::new(lsb)?,
            ),
        ])
    }

    fn send_key(&self, key: &str, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        let note = key_to_midi_note(key);
        Ok(vec![
            self.deck_select_message(config)?,
            MidiMessage::ControlChange(
                config.key_channel,
                ControlNumber::new(DENON_KEY_CC)?,
                MidiValue::new(note)?,
            ),
        ])
    }

    fn send_energy(&self, energy: u8, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        Ok(vec![
            self.deck_select_message(config)?,
            MidiMessage::ControlChange(
                config.energy_channel,
                ControlNumber::new(config.energy_cc)?,
                MidiValue::from_energy(energy),
            ),
        ])
    }

    fn send_track_info(
        &self,
        info: &TrackInfo,
        config: &MidiConfig,
    ) -> Result<Vec<MidiMessage>, MidiError> {
        let mut messages = Vec::new();
        messages.extend(self.send_bpm(info.bpm, config)?);
        messages.extend(self.send_key(&info.key, config)?);
        messages.extend(self.send_energy(info.energy, config)?);
        messages.push(self.build_sysex_track_info(info));
        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> MidiConfig {
        MidiConfig::default()
    }

    #[test]
    fn encode_bpm_128() {
        let (msb, lsb) = DenonProtocol::encode_bpm(128.0);
        let reconstructed = ((msb as u16) << 7) | (lsb as u16);
        assert_eq!(reconstructed, 12800);
    }

    #[test]
    fn send_bpm_includes_deck_select() {
        let protocol = DenonProtocol::with_deck(1);
        let config = default_config();
        let messages = protocol.send_bpm(128.0, &config).unwrap();
        // deck select + MSB + LSB = 3
        assert_eq!(messages.len(), 3);
        // First message is deck select
        let bytes = messages[0].to_bytes();
        assert_eq!(bytes[1], DENON_DECK_SELECT_CC);
        assert_eq!(bytes[2], 1); // Deck 1
    }

    #[test]
    fn send_key_includes_deck_select() {
        let protocol = DenonProtocol::with_deck(2);
        let config = default_config();
        let messages = protocol.send_key("Cm", &config).unwrap();
        assert_eq!(messages.len(), 2);
        // Deck select
        let bytes0 = messages[0].to_bytes();
        assert_eq!(bytes0[2], 2); // Deck 2
        // Key CC
        let bytes1 = messages[1].to_bytes();
        assert_eq!(bytes1[1], DENON_KEY_CC);
    }

    #[test]
    fn deck_clamped_to_3() {
        let protocol = DenonProtocol::with_deck(10);
        assert_eq!(protocol.active_deck, 3);
    }

    #[test]
    fn sysex_contains_deck_and_data() {
        let protocol = DenonProtocol::with_deck(1);
        let info = TrackInfo {
            bpm: 128.0,
            key: "Am".to_string(),
            energy: 80,
            title: "Test".to_string(),
            artist: "DJ".to_string(),
        };
        let msg = protocol.build_sysex_track_info(&info);
        let bytes = msg.to_bytes();
        assert_eq!(bytes[0], 0xF0);
        assert_eq!(&bytes[1..4], &DENON_SYSEX_ID);
        assert_eq!(bytes[4], 0x20); // Command
        assert_eq!(bytes[5], 1); // Deck 1
        assert_eq!(bytes[bytes.len() - 1], 0xF7);
    }

    #[test]
    fn protocol_name() {
        assert_eq!(DenonProtocol::new().name(), "Denon SC6000");
    }

    #[test]
    fn send_track_info_full() {
        let protocol = DenonProtocol::new();
        let config = default_config();
        let info = TrackInfo {
            bpm: 140.0,
            key: "F".to_string(),
            energy: 90,
            title: "Track".to_string(),
            artist: "Artist".to_string(),
        };
        let messages = protocol.send_track_info(&info, &config).unwrap();
        // 3 (bpm) + 2 (key) + 2 (energy) + 1 (sysex) = 8
        assert_eq!(messages.len(), 8);
    }
}
