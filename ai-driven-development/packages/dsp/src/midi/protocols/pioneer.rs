//! Pioneer CDJ-2000NXS2 / CDJ-3000 / DJM protocol handler.
//!
//! Pioneer DJ equipment uses a combination of standard MIDI CC messages
//! and SysEx for metadata exchange. The CDJ-3000 uses PRO DJ LINK protocol
//! over Ethernet, but also accepts MIDI for BPM/key display.
//!
//! Pioneer SysEx Manufacturer ID: 0x00 0x40 0x05 (Pioneer Corporation)

use super::{key_to_midi_note, ProtocolHandler};
use crate::midi::types::{
    ControlNumber, MidiConfig, MidiError, MidiMessage, MidiValue, TrackInfo,
};

/// Pioneer SysEx manufacturer ID.
const PIONEER_SYSEX_ID: [u8; 3] = [0x00, 0x40, 0x05];

/// Pioneer-specific CC assignments for CDJ feedback.
const PIONEER_BPM_CC_MSB: u8 = 46; // BPM coarse
const PIONEER_BPM_CC_LSB: u8 = 47; // BPM fine
const PIONEER_KEY_CC: u8 = 48; // Musical key

/// Protocol handler for Pioneer CDJ/DJM equipment.
pub struct PioneerProtocol;

impl PioneerProtocol {
    pub fn new() -> Self {
        Self
    }

    /// Encode BPM as two 7-bit MIDI values (MSB/LSB).
    /// BPM * 100 to preserve decimal (e.g., 128.50 -> 12850).
    fn encode_bpm(bpm: f64) -> (u8, u8) {
        let bpm_int = (bpm * 100.0).round() as u16;
        let msb = ((bpm_int >> 7) & 0x7F) as u8;
        let lsb = (bpm_int & 0x7F) as u8;
        (msb, lsb)
    }

    /// Build a Pioneer SysEx message for track metadata.
    fn build_sysex_track_info(info: &TrackInfo) -> MidiMessage {
        let mut data = Vec::new();
        // Manufacturer ID
        data.extend_from_slice(&PIONEER_SYSEX_ID);
        // Command: track info update (0x10)
        data.push(0x10);
        // BPM as 14-bit value
        let (msb, lsb) = Self::encode_bpm(info.bpm);
        data.push(msb);
        data.push(lsb);
        // Key as note number
        data.push(key_to_midi_note(&info.key));
        // Energy
        data.push(info.energy.min(127));
        MidiMessage::SysEx(data)
    }
}

impl ProtocolHandler for PioneerProtocol {
    fn name(&self) -> &str {
        "Pioneer CDJ/DJM"
    }

    fn send_bpm(&self, bpm: f64, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        let (msb, lsb) = Self::encode_bpm(bpm);
        Ok(vec![
            MidiMessage::ControlChange(
                config.bpm_channel,
                ControlNumber::new(PIONEER_BPM_CC_MSB)?,
                MidiValue::new(msb)?,
            ),
            MidiMessage::ControlChange(
                config.bpm_channel,
                ControlNumber::new(PIONEER_BPM_CC_LSB)?,
                MidiValue::new(lsb)?,
            ),
        ])
    }

    fn send_key(&self, key: &str, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        let note = key_to_midi_note(key);
        Ok(vec![MidiMessage::ControlChange(
            config.key_channel,
            ControlNumber::new(PIONEER_KEY_CC)?,
            MidiValue::new(note)?,
        )])
    }

    fn send_energy(&self, energy: u8, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        Ok(vec![MidiMessage::ControlChange(
            config.energy_channel,
            ControlNumber::new(config.energy_cc)?,
            MidiValue::from_energy(energy),
        )])
    }

    fn send_track_info(
        &self,
        info: &TrackInfo,
        config: &MidiConfig,
    ) -> Result<Vec<MidiMessage>, MidiError> {
        let mut messages = Vec::new();
        // Send standard CC messages
        messages.extend(self.send_bpm(info.bpm, config)?);
        messages.extend(self.send_key(&info.key, config)?);
        messages.extend(self.send_energy(info.energy, config)?);
        // Also send SysEx with full track info
        messages.push(Self::build_sysex_track_info(info));
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
        let (msb, lsb) = PioneerProtocol::encode_bpm(128.0);
        let reconstructed = ((msb as u16) << 7) | (lsb as u16);
        assert_eq!(reconstructed, 12800);
    }

    #[test]
    fn encode_bpm_128_5() {
        let (msb, lsb) = PioneerProtocol::encode_bpm(128.5);
        let reconstructed = ((msb as u16) << 7) | (lsb as u16);
        assert_eq!(reconstructed, 12850);
    }

    #[test]
    fn send_bpm_generates_two_cc() {
        let protocol = PioneerProtocol::new();
        let config = default_config();
        let messages = protocol.send_bpm(128.0, &config).unwrap();
        assert_eq!(messages.len(), 2);
        // MSB CC
        let bytes0 = messages[0].to_bytes();
        assert_eq!(bytes0[0], 0xB0); // CC on channel 0
        assert_eq!(bytes0[1], PIONEER_BPM_CC_MSB);
        // LSB CC
        let bytes1 = messages[1].to_bytes();
        assert_eq!(bytes1[1], PIONEER_BPM_CC_LSB);
    }

    #[test]
    fn send_key_am() {
        let protocol = PioneerProtocol::new();
        let config = default_config();
        let messages = protocol.send_key("Am", &config).unwrap();
        assert_eq!(messages.len(), 1);
        let bytes = messages[0].to_bytes();
        assert_eq!(bytes[1], PIONEER_KEY_CC);
        assert_eq!(bytes[2], 7); // Am = 8A = note 7
    }

    #[test]
    fn send_energy() {
        let protocol = PioneerProtocol::new();
        let config = default_config();
        let messages = protocol.send_energy(75, &config).unwrap();
        assert_eq!(messages.len(), 1);
        let bytes = messages[0].to_bytes();
        assert_eq!(bytes[2], MidiValue::from_energy(75).value());
    }

    #[test]
    fn send_track_info_includes_sysex() {
        let protocol = PioneerProtocol::new();
        let config = default_config();
        let info = TrackInfo {
            bpm: 128.0,
            key: "Am".to_string(),
            energy: 80,
            title: "Test".to_string(),
            artist: "DJ".to_string(),
        };
        let messages = protocol.send_track_info(&info, &config).unwrap();
        // Should have: 2 BPM CCs + 1 key CC + 1 energy CC + 1 SysEx = 5
        assert_eq!(messages.len(), 5);
        // Last message should be SysEx
        let last = messages.last().unwrap();
        let bytes = last.to_bytes();
        assert_eq!(bytes[0], 0xF0); // SysEx start
        assert_eq!(bytes[bytes.len() - 1], 0xF7); // SysEx end
        // Check manufacturer ID
        assert_eq!(&bytes[1..4], &PIONEER_SYSEX_ID);
    }

    #[test]
    fn sysex_contains_track_data() {
        let info = TrackInfo {
            bpm: 130.0,
            key: "Cm".to_string(),
            energy: 90,
            title: "Track".to_string(),
            artist: "Artist".to_string(),
        };
        let msg = PioneerProtocol::build_sysex_track_info(&info);
        let bytes = msg.to_bytes();
        // F0 [manufacturer 3 bytes] [cmd 1] [bpm_msb 1] [bpm_lsb 1] [key 1] [energy 1] F7
        assert_eq!(bytes.len(), 10);
        assert_eq!(bytes[4], 0x10); // Command byte
    }

    #[test]
    fn protocol_name() {
        assert_eq!(PioneerProtocol::new().name(), "Pioneer CDJ/DJM");
    }
}
