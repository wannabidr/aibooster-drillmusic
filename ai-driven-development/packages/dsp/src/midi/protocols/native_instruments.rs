//! Native Instruments Traktor controller protocol handler.
//!
//! NI controllers (Traktor Kontrol S2, S4, D2, etc.) use standard MIDI
//! CC messages. Traktor mapping files (.tsi) define CC-to-parameter
//! assignments. This handler uses a configurable CC mapping to support
//! multiple controller models.

use super::{key_to_midi_note, ProtocolHandler};
use crate::midi::types::{
    ControlNumber, MidiConfig, MidiError, MidiMessage, MidiValue,
};

/// Default CC mappings for NI Traktor controllers.
/// These match common Traktor MIDI mapping conventions.
const NI_BPM_CC: u8 = 20;
const NI_KEY_CC: u8 = 21;
const NI_ENERGY_CC: u8 = 22;
const NI_DECK_CC: u8 = 23;

/// Configurable CC mapping for NI controllers.
#[derive(Debug, Clone)]
pub struct NiMapping {
    pub bpm_cc: u8,
    pub key_cc: u8,
    pub energy_cc: u8,
    pub deck_cc: u8,
}

impl Default for NiMapping {
    fn default() -> Self {
        Self {
            bpm_cc: NI_BPM_CC,
            key_cc: NI_KEY_CC,
            energy_cc: NI_ENERGY_CC,
            deck_cc: NI_DECK_CC,
        }
    }
}

/// Protocol handler for Native Instruments controllers.
pub struct NiProtocol {
    mapping: NiMapping,
    active_deck: u8,
}

impl NiProtocol {
    pub fn new() -> Self {
        Self {
            mapping: NiMapping::default(),
            active_deck: 0,
        }
    }

    pub fn with_mapping(mapping: NiMapping) -> Self {
        Self {
            mapping,
            active_deck: 0,
        }
    }

    pub fn with_deck(mut self, deck: u8) -> Self {
        self.active_deck = deck.min(3);
        self
    }

    /// Build a deck select CC message.
    fn deck_select_message(&self, config: &MidiConfig) -> Result<MidiMessage, MidiError> {
        Ok(MidiMessage::ControlChange(
            config.bpm_channel,
            ControlNumber::new(self.mapping.deck_cc)?,
            MidiValue::new(self.active_deck)?,
        ))
    }
}

impl ProtocolHandler for NiProtocol {
    fn name(&self) -> &str {
        "Native Instruments"
    }

    fn send_bpm(&self, bpm: f64, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        Ok(vec![
            self.deck_select_message(config)?,
            MidiMessage::ControlChange(
                config.bpm_channel,
                ControlNumber::new(self.mapping.bpm_cc)?,
                MidiValue::from_bpm(bpm),
            ),
        ])
    }

    fn send_key(&self, key: &str, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        let note = key_to_midi_note(key);
        Ok(vec![
            self.deck_select_message(config)?,
            MidiMessage::ControlChange(
                config.key_channel,
                ControlNumber::new(self.mapping.key_cc)?,
                MidiValue::new(note)?,
            ),
        ])
    }

    fn send_energy(&self, energy: u8, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        Ok(vec![
            self.deck_select_message(config)?,
            MidiMessage::ControlChange(
                config.energy_channel,
                ControlNumber::new(self.mapping.energy_cc)?,
                MidiValue::from_energy(energy),
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::TrackInfo;

    fn default_config() -> MidiConfig {
        MidiConfig::default()
    }

    #[test]
    fn default_mapping() {
        let mapping = NiMapping::default();
        assert_eq!(mapping.bpm_cc, 20);
        assert_eq!(mapping.key_cc, 21);
        assert_eq!(mapping.energy_cc, 22);
        assert_eq!(mapping.deck_cc, 23);
    }

    #[test]
    fn send_bpm_with_deck_select() {
        let protocol = NiProtocol::new().with_deck(1);
        let config = default_config();
        let messages = protocol.send_bpm(128.0, &config).unwrap();
        assert_eq!(messages.len(), 2);
        // Deck select
        let bytes0 = messages[0].to_bytes();
        assert_eq!(bytes0[1], NI_DECK_CC);
        assert_eq!(bytes0[2], 1);
        // BPM CC
        let bytes1 = messages[1].to_bytes();
        assert_eq!(bytes1[1], NI_BPM_CC);
    }

    #[test]
    fn send_key() {
        let protocol = NiProtocol::new();
        let config = default_config();
        let messages = protocol.send_key("Am", &config).unwrap();
        assert_eq!(messages.len(), 2);
        let bytes1 = messages[1].to_bytes();
        assert_eq!(bytes1[1], NI_KEY_CC);
        assert_eq!(bytes1[2], 7); // Am = 8A = 7
    }

    #[test]
    fn send_energy() {
        let protocol = NiProtocol::new();
        let config = default_config();
        let messages = protocol.send_energy(50, &config).unwrap();
        assert_eq!(messages.len(), 2);
        let bytes1 = messages[1].to_bytes();
        assert_eq!(bytes1[1], NI_ENERGY_CC);
        assert_eq!(bytes1[2], MidiValue::from_energy(50).value());
    }

    #[test]
    fn custom_mapping() {
        let mapping = NiMapping {
            bpm_cc: 30,
            key_cc: 31,
            energy_cc: 32,
            deck_cc: 33,
        };
        let protocol = NiProtocol::with_mapping(mapping);
        let config = default_config();
        let messages = protocol.send_bpm(128.0, &config).unwrap();
        let bytes0 = messages[0].to_bytes();
        assert_eq!(bytes0[1], 33); // Custom deck CC
        let bytes1 = messages[1].to_bytes();
        assert_eq!(bytes1[1], 30); // Custom BPM CC
    }

    #[test]
    fn send_track_info_default() {
        let protocol = NiProtocol::new();
        let config = default_config();
        let info = TrackInfo {
            bpm: 174.0,
            key: "Fm".to_string(),
            energy: 95,
            title: "DnB Track".to_string(),
            artist: "Producer".to_string(),
        };
        let messages = protocol.send_track_info(&info, &config).unwrap();
        // 2 (bpm) + 2 (key) + 2 (energy) = 6
        assert_eq!(messages.len(), 6);
    }

    #[test]
    fn protocol_name() {
        assert_eq!(NiProtocol::new().name(), "Native Instruments");
    }

    #[test]
    fn deck_clamped() {
        let protocol = NiProtocol::new().with_deck(10);
        assert_eq!(protocol.active_deck, 3);
    }
}
