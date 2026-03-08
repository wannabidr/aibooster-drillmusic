//! MIDI domain types for DJ hardware communication.

use serde::{Deserialize, Serialize};

/// Represents a connected MIDI device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiDevice {
    /// Unique port identifier (from midir).
    pub port_id: String,
    /// Human-readable device name.
    pub name: String,
    /// Detected hardware type.
    pub device_type: DeviceType,
    /// Whether the device supports input (receiving from hardware).
    pub has_input: bool,
    /// Whether the device supports output (sending to hardware).
    pub has_output: bool,
}

/// Known DJ hardware types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Pioneer CDJ-2000NXS2 or CDJ-3000.
    PioneerCdj,
    /// Pioneer DJM mixer.
    PioneerDjm,
    /// Denon SC6000 / SC6000M.
    DenonSc6000,
    /// Native Instruments Traktor controller (S2, S4, D2, etc.).
    NativeInstruments,
    /// Generic MIDI controller.
    Generic,
}

impl DeviceType {
    /// Detect device type from its name string.
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("cdj") || lower.contains("xdj") {
            DeviceType::PioneerCdj
        } else if lower.contains("djm") {
            DeviceType::PioneerDjm
        } else if lower.contains("sc6000") || lower.contains("sc5000") || lower.contains("denon") {
            DeviceType::DenonSc6000
        } else if lower.contains("traktor")
            || lower.contains("kontrol")
            || lower.contains("native instruments")
            || lower.contains("maschine")
        {
            DeviceType::NativeInstruments
        } else {
            DeviceType::Generic
        }
    }
}

/// Track metadata to send to DJ hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackInfo {
    /// BPM value (e.g., 128.0).
    pub bpm: f64,
    /// Musical key as string (e.g., "Am", "8B").
    pub key: String,
    /// Energy level (0-100).
    pub energy: u8,
    /// Track title.
    pub title: String,
    /// Artist name.
    pub artist: String,
}

/// MIDI channel (0-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MidiChannel(u8);

impl MidiChannel {
    pub fn new(channel: u8) -> Result<Self, MidiError> {
        if channel > 15 {
            return Err(MidiError::InvalidChannel(channel));
        }
        Ok(Self(channel))
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

impl Default for MidiChannel {
    fn default() -> Self {
        Self(0)
    }
}

/// MIDI Control Change number (0-127).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlNumber(u8);

impl ControlNumber {
    pub fn new(cc: u8) -> Result<Self, MidiError> {
        if cc > 127 {
            return Err(MidiError::InvalidControlNumber(cc));
        }
        Ok(Self(cc))
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

/// MIDI value (0-127).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MidiValue(u8);

impl MidiValue {
    pub fn new(val: u8) -> Result<Self, MidiError> {
        if val > 127 {
            return Err(MidiError::InvalidValue(val));
        }
        Ok(Self(val))
    }

    pub fn value(&self) -> u8 {
        self.0
    }

    /// Scale a float value (0.0-1.0) to MIDI range (0-127).
    pub fn from_normalized(val: f64) -> Self {
        let clamped = val.clamp(0.0, 1.0);
        Self((clamped * 127.0).round() as u8)
    }

    /// Scale a BPM value to MIDI range. Maps 60-200 BPM to 0-127.
    pub fn from_bpm(bpm: f64) -> Self {
        let normalized = (bpm - 60.0) / 140.0; // 60-200 range
        Self::from_normalized(normalized)
    }

    /// Scale an energy value (0-100) to MIDI range (0-127).
    pub fn from_energy(energy: u8) -> Self {
        Self((energy as f64 / 100.0 * 127.0).round() as u8)
    }
}

/// Outgoing MIDI message types.
#[derive(Debug, Clone)]
pub enum MidiMessage {
    /// Control Change: channel, CC number, value.
    ControlChange(MidiChannel, ControlNumber, MidiValue),
    /// Note On: channel, note, velocity.
    NoteOn(MidiChannel, u8, u8),
    /// Note Off: channel, note, velocity.
    NoteOff(MidiChannel, u8, u8),
    /// System Exclusive data (vendor-specific).
    SysEx(Vec<u8>),
    /// MIDI Clock tick (0xF8).
    Clock,
    /// MIDI Start (0xFA).
    Start,
    /// MIDI Stop (0xFC).
    Stop,
}

impl MidiMessage {
    /// Serialize to raw MIDI bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            MidiMessage::ControlChange(ch, cc, val) => {
                vec![0xB0 | ch.value(), cc.value(), val.value()]
            }
            MidiMessage::NoteOn(ch, note, vel) => {
                vec![0x90 | ch.value(), note & 0x7F, vel & 0x7F]
            }
            MidiMessage::NoteOff(ch, note, vel) => {
                vec![0x80 | ch.value(), note & 0x7F, vel & 0x7F]
            }
            MidiMessage::SysEx(data) => {
                let mut bytes = Vec::with_capacity(data.len() + 2);
                bytes.push(0xF0);
                bytes.extend_from_slice(data);
                bytes.push(0xF7);
                bytes
            }
            MidiMessage::Clock => vec![0xF8],
            MidiMessage::Start => vec![0xFA],
            MidiMessage::Stop => vec![0xFC],
        }
    }
}

/// Configuration for MIDI output behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiConfig {
    /// MIDI channel for BPM output (default: 0).
    pub bpm_channel: MidiChannel,
    /// CC number for BPM output (default: 1).
    pub bpm_cc: u8,
    /// MIDI channel for key output (default: 0).
    pub key_channel: MidiChannel,
    /// CC number for key output (default: 2).
    pub key_cc: u8,
    /// MIDI channel for energy output (default: 0).
    pub energy_channel: MidiChannel,
    /// CC number for energy output (default: 3).
    pub energy_cc: u8,
    /// Whether to send MIDI clock sync.
    pub send_clock: bool,
}

impl Default for MidiConfig {
    fn default() -> Self {
        Self {
            bpm_channel: MidiChannel::default(),
            bpm_cc: 1,
            key_channel: MidiChannel::default(),
            key_cc: 2,
            energy_channel: MidiChannel::default(),
            energy_cc: 3,
            send_clock: false,
        }
    }
}

/// Errors from MIDI operations.
#[derive(Debug, Clone)]
pub enum MidiError {
    /// No MIDI output ports available.
    NoOutputPorts,
    /// Specified port not found.
    PortNotFound(String),
    /// Failed to connect to MIDI port.
    ConnectionFailed(String),
    /// Failed to send MIDI message.
    SendFailed(String),
    /// MIDI channel out of range (0-15).
    InvalidChannel(u8),
    /// CC number out of range (0-127).
    InvalidControlNumber(u8),
    /// Value out of range (0-127).
    InvalidValue(u8),
    /// Device already connected.
    AlreadyConnected(String),
    /// Not connected to any device.
    NotConnected,
    /// Protocol error from hardware.
    ProtocolError(String),
}

impl std::fmt::Display for MidiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoOutputPorts => write!(f, "no MIDI output ports available"),
            Self::PortNotFound(name) => write!(f, "MIDI port not found: '{name}'"),
            Self::ConnectionFailed(msg) => write!(f, "MIDI connection failed: {msg}"),
            Self::SendFailed(msg) => write!(f, "MIDI send failed: {msg}"),
            Self::InvalidChannel(ch) => write!(f, "invalid MIDI channel: {ch} (must be 0-15)"),
            Self::InvalidControlNumber(cc) => {
                write!(f, "invalid CC number: {cc} (must be 0-127)")
            }
            Self::InvalidValue(v) => write!(f, "invalid MIDI value: {v} (must be 0-127)"),
            Self::AlreadyConnected(name) => write!(f, "already connected to: {name}"),
            Self::NotConnected => write!(f, "not connected to any MIDI device"),
            Self::ProtocolError(msg) => write!(f, "protocol error: {msg}"),
        }
    }
}

impl std::error::Error for MidiError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midi_channel_valid_range() {
        for ch in 0..=15 {
            assert!(MidiChannel::new(ch).is_ok());
        }
        assert!(MidiChannel::new(16).is_err());
    }

    #[test]
    fn control_number_valid_range() {
        assert!(ControlNumber::new(0).is_ok());
        assert!(ControlNumber::new(127).is_ok());
        assert!(ControlNumber::new(128).is_err());
    }

    #[test]
    fn midi_value_valid_range() {
        assert!(MidiValue::new(0).is_ok());
        assert!(MidiValue::new(127).is_ok());
        assert!(MidiValue::new(128).is_err());
    }

    #[test]
    fn midi_value_from_normalized() {
        assert_eq!(MidiValue::from_normalized(0.0).value(), 0);
        assert_eq!(MidiValue::from_normalized(1.0).value(), 127);
        assert_eq!(MidiValue::from_normalized(0.5).value(), 64);
        // Clamping
        assert_eq!(MidiValue::from_normalized(-0.5).value(), 0);
        assert_eq!(MidiValue::from_normalized(1.5).value(), 127);
    }

    #[test]
    fn midi_value_from_bpm() {
        // 60 BPM -> 0, 200 BPM -> 127
        assert_eq!(MidiValue::from_bpm(60.0).value(), 0);
        assert_eq!(MidiValue::from_bpm(200.0).value(), 127);
        // 130 BPM -> ~63
        let mid = MidiValue::from_bpm(130.0).value();
        assert!(mid >= 62 && mid <= 65);
    }

    #[test]
    fn midi_value_from_energy() {
        assert_eq!(MidiValue::from_energy(0).value(), 0);
        assert_eq!(MidiValue::from_energy(100).value(), 127);
        assert_eq!(MidiValue::from_energy(50).value(), 64);
    }

    #[test]
    fn device_type_detection() {
        assert_eq!(DeviceType::from_name("Pioneer CDJ-3000"), DeviceType::PioneerCdj);
        assert_eq!(DeviceType::from_name("XDJ-RX3"), DeviceType::PioneerCdj);
        assert_eq!(DeviceType::from_name("DJM-900NXS2"), DeviceType::PioneerDjm);
        assert_eq!(DeviceType::from_name("Denon SC6000"), DeviceType::DenonSc6000);
        assert_eq!(
            DeviceType::from_name("Traktor Kontrol S4"),
            DeviceType::NativeInstruments
        );
        assert_eq!(
            DeviceType::from_name("Native Instruments S2"),
            DeviceType::NativeInstruments
        );
        assert_eq!(DeviceType::from_name("Random Controller"), DeviceType::Generic);
    }

    #[test]
    fn control_change_message_bytes() {
        let msg = MidiMessage::ControlChange(
            MidiChannel::new(0).unwrap(),
            ControlNumber::new(1).unwrap(),
            MidiValue::new(64).unwrap(),
        );
        assert_eq!(msg.to_bytes(), vec![0xB0, 0x01, 0x40]);
    }

    #[test]
    fn control_change_channel_3() {
        let msg = MidiMessage::ControlChange(
            MidiChannel::new(3).unwrap(),
            ControlNumber::new(7).unwrap(),
            MidiValue::new(127).unwrap(),
        );
        assert_eq!(msg.to_bytes(), vec![0xB3, 0x07, 0x7F]);
    }

    #[test]
    fn note_on_message_bytes() {
        let msg = MidiMessage::NoteOn(MidiChannel::new(0).unwrap(), 60, 100);
        assert_eq!(msg.to_bytes(), vec![0x90, 60, 100]);
    }

    #[test]
    fn note_off_message_bytes() {
        let msg = MidiMessage::NoteOff(MidiChannel::new(0).unwrap(), 60, 0);
        assert_eq!(msg.to_bytes(), vec![0x80, 60, 0]);
    }

    #[test]
    fn sysex_message_bytes() {
        let msg = MidiMessage::SysEx(vec![0x7E, 0x06, 0x01]);
        assert_eq!(msg.to_bytes(), vec![0xF0, 0x7E, 0x06, 0x01, 0xF7]);
    }

    #[test]
    fn clock_message_bytes() {
        assert_eq!(MidiMessage::Clock.to_bytes(), vec![0xF8]);
    }

    #[test]
    fn start_message_bytes() {
        assert_eq!(MidiMessage::Start.to_bytes(), vec![0xFA]);
    }

    #[test]
    fn stop_message_bytes() {
        assert_eq!(MidiMessage::Stop.to_bytes(), vec![0xFC]);
    }

    #[test]
    fn default_config() {
        let config = MidiConfig::default();
        assert_eq!(config.bpm_channel.value(), 0);
        assert_eq!(config.bpm_cc, 1);
        assert_eq!(config.key_cc, 2);
        assert_eq!(config.energy_cc, 3);
        assert!(!config.send_clock);
    }

    #[test]
    fn track_info_serialization() {
        let info = TrackInfo {
            bpm: 128.0,
            key: "Am".to_string(),
            energy: 75,
            title: "Test Track".to_string(),
            artist: "Test Artist".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: TrackInfo = serde_json::from_str(&json).unwrap();
        assert!((deserialized.bpm - 128.0).abs() < f64::EPSILON);
        assert_eq!(deserialized.key, "Am");
        assert_eq!(deserialized.energy, 75);
    }
}
