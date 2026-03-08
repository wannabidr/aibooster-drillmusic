//! MIDI engine: device discovery, connection management, and message sending.

use midir::{MidiOutput, MidiOutputConnection, MidiOutputPort};

use super::protocols::denon::DenonProtocol;
use super::protocols::native_instruments::NiProtocol;
use super::protocols::pioneer::PioneerProtocol;
use super::protocols::ProtocolHandler;
use super::types::{DeviceType, MidiConfig, MidiDevice, MidiError, MidiMessage, TrackInfo};

/// The MIDI engine manages device discovery and output connections.
pub struct MidiEngine {
    config: MidiConfig,
    connection: Option<ActiveConnection>,
}

/// An active MIDI output connection with its protocol handler.
struct ActiveConnection {
    conn: MidiOutputConnection,
    device: MidiDevice,
    protocol: Box<dyn ProtocolHandler>,
}

impl MidiEngine {
    /// Create a new MIDI engine with default configuration.
    pub fn new() -> Self {
        Self {
            config: MidiConfig::default(),
            connection: None,
        }
    }

    /// Create a new MIDI engine with custom configuration.
    pub fn with_config(config: MidiConfig) -> Self {
        Self {
            config,
            connection: None,
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MidiConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: MidiConfig) {
        self.config = config;
    }

    /// Scan for available MIDI output devices.
    pub fn list_devices(&self) -> Result<Vec<MidiDevice>, MidiError> {
        let midi_out = MidiOutput::new("AI DJ Assist Scanner")
            .map_err(|e| MidiError::ConnectionFailed(e.to_string()))?;

        let ports = midi_out.ports();
        let mut devices = Vec::new();

        for port in &ports {
            let name = midi_out
                .port_name(port)
                .unwrap_or_else(|_| "Unknown".to_string());
            let device_type = DeviceType::from_name(&name);
            let port_id = name.clone();

            devices.push(MidiDevice {
                port_id,
                name,
                device_type,
                has_input: false, // We only handle output for now
                has_output: true,
            });
        }

        Ok(devices)
    }

    /// Connect to a MIDI output device by name.
    pub fn connect(&mut self, device_name: &str) -> Result<MidiDevice, MidiError> {
        if let Some(ref conn) = self.connection {
            return Err(MidiError::AlreadyConnected(conn.device.name.clone()));
        }

        let midi_out = MidiOutput::new("AI DJ Assist")
            .map_err(|e| MidiError::ConnectionFailed(e.to_string()))?;

        let ports = midi_out.ports();
        let (port, name) = self.find_port(&midi_out, &ports, device_name)?;

        let device_type = DeviceType::from_name(&name);
        let port_id = name.clone();

        let conn = midi_out
            .connect(port, "AI DJ Assist Output")
            .map_err(|e| MidiError::ConnectionFailed(e.to_string()))?;

        let device = MidiDevice {
            port_id,
            name: name.clone(),
            device_type,
            has_input: false,
            has_output: true,
        };

        let protocol: Box<dyn ProtocolHandler> = match device_type {
            DeviceType::PioneerCdj | DeviceType::PioneerDjm => {
                Box::new(PioneerProtocol::new())
            }
            DeviceType::DenonSc6000 => Box::new(DenonProtocol::new()),
            DeviceType::NativeInstruments => Box::new(NiProtocol::new()),
            DeviceType::Generic => Box::new(GenericProtocol),
        };

        let device_clone = device.clone();
        self.connection = Some(ActiveConnection {
            conn,
            device,
            protocol,
        });

        Ok(device_clone)
    }

    /// Connect to a MIDI output device by port index.
    pub fn connect_by_index(&mut self, index: usize) -> Result<MidiDevice, MidiError> {
        if let Some(ref conn) = self.connection {
            return Err(MidiError::AlreadyConnected(conn.device.name.clone()));
        }

        let midi_out = MidiOutput::new("AI DJ Assist")
            .map_err(|e| MidiError::ConnectionFailed(e.to_string()))?;

        let ports = midi_out.ports();
        if index >= ports.len() {
            return Err(MidiError::PortNotFound(format!("index {index}")));
        }

        let port = &ports[index];
        let name = midi_out
            .port_name(port)
            .unwrap_or_else(|_| "Unknown".to_string());
        let device_type = DeviceType::from_name(&name);
        let port_id = name.clone();

        let conn = midi_out
            .connect(port, "AI DJ Assist Output")
            .map_err(|e| MidiError::ConnectionFailed(e.to_string()))?;

        let device = MidiDevice {
            port_id,
            name,
            device_type,
            has_input: false,
            has_output: true,
        };

        let protocol: Box<dyn ProtocolHandler> = match device_type {
            DeviceType::PioneerCdj | DeviceType::PioneerDjm => {
                Box::new(PioneerProtocol::new())
            }
            DeviceType::DenonSc6000 => Box::new(DenonProtocol::new()),
            DeviceType::NativeInstruments => Box::new(NiProtocol::new()),
            DeviceType::Generic => Box::new(GenericProtocol),
        };

        let device_clone = device.clone();
        self.connection = Some(ActiveConnection {
            conn,
            device,
            protocol,
        });

        Ok(device_clone)
    }

    /// Disconnect from the current device.
    pub fn disconnect(&mut self) -> Result<(), MidiError> {
        match self.connection.take() {
            Some(conn) => {
                conn.conn.close();
                Ok(())
            }
            None => Err(MidiError::NotConnected),
        }
    }

    /// Check if currently connected to a device.
    pub fn is_connected(&self) -> bool {
        self.connection.is_some()
    }

    /// Get the currently connected device info.
    pub fn connected_device(&self) -> Option<&MidiDevice> {
        self.connection.as_ref().map(|c| &c.device)
    }

    /// Send a raw MIDI message to the connected device.
    pub fn send_message(&mut self, msg: &MidiMessage) -> Result<(), MidiError> {
        let conn = self
            .connection
            .as_mut()
            .ok_or(MidiError::NotConnected)?;

        let bytes = msg.to_bytes();
        conn.conn
            .send(&bytes)
            .map_err(|e| MidiError::SendFailed(e.to_string()))?;
        Ok(())
    }

    /// Send multiple MIDI messages to the connected device.
    pub fn send_messages(&mut self, messages: &[MidiMessage]) -> Result<(), MidiError> {
        for msg in messages {
            self.send_message(msg)?;
        }
        Ok(())
    }

    /// Send track info to the connected device using the appropriate protocol.
    pub fn send_track_info(&mut self, info: &TrackInfo) -> Result<(), MidiError> {
        let conn = self
            .connection
            .as_ref()
            .ok_or(MidiError::NotConnected)?;

        let messages = conn.protocol.send_track_info(info, &self.config)?;

        let conn = self
            .connection
            .as_mut()
            .ok_or(MidiError::NotConnected)?;

        for msg in &messages {
            let bytes = msg.to_bytes();
            conn.conn
                .send(&bytes)
                .map_err(|e| MidiError::SendFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Send BPM to the connected device using the appropriate protocol.
    pub fn send_bpm(&mut self, bpm: f64) -> Result<(), MidiError> {
        let conn = self
            .connection
            .as_ref()
            .ok_or(MidiError::NotConnected)?;

        let messages = conn.protocol.send_bpm(bpm, &self.config)?;

        let conn = self
            .connection
            .as_mut()
            .ok_or(MidiError::NotConnected)?;

        for msg in &messages {
            let bytes = msg.to_bytes();
            conn.conn
                .send(&bytes)
                .map_err(|e| MidiError::SendFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Send key to the connected device using the appropriate protocol.
    pub fn send_key(&mut self, key: &str) -> Result<(), MidiError> {
        let conn = self
            .connection
            .as_ref()
            .ok_or(MidiError::NotConnected)?;

        let messages = conn.protocol.send_key(key, &self.config)?;

        let conn = self
            .connection
            .as_mut()
            .ok_or(MidiError::NotConnected)?;

        for msg in &messages {
            let bytes = msg.to_bytes();
            conn.conn
                .send(&bytes)
                .map_err(|e| MidiError::SendFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Send energy to the connected device using the appropriate protocol.
    pub fn send_energy(&mut self, energy: u8) -> Result<(), MidiError> {
        let conn = self
            .connection
            .as_ref()
            .ok_or(MidiError::NotConnected)?;

        let messages = conn.protocol.send_energy(energy, &self.config)?;

        let conn = self
            .connection
            .as_mut()
            .ok_or(MidiError::NotConnected)?;

        for msg in &messages {
            let bytes = msg.to_bytes();
            conn.conn
                .send(&bytes)
                .map_err(|e| MidiError::SendFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Send a MIDI clock tick.
    pub fn send_clock(&mut self) -> Result<(), MidiError> {
        self.send_message(&MidiMessage::Clock)
    }

    /// Send MIDI start message.
    pub fn send_start(&mut self) -> Result<(), MidiError> {
        self.send_message(&MidiMessage::Start)
    }

    /// Send MIDI stop message.
    pub fn send_stop(&mut self) -> Result<(), MidiError> {
        self.send_message(&MidiMessage::Stop)
    }

    /// Get the protocol name for the connected device.
    pub fn protocol_name(&self) -> Option<&str> {
        self.connection.as_ref().map(|c| c.protocol.name())
    }

    /// Find a MIDI output port by name (case-insensitive partial match).
    fn find_port<'a>(
        &self,
        midi_out: &MidiOutput,
        ports: &'a [MidiOutputPort],
        name: &str,
    ) -> Result<(&'a MidiOutputPort, String), MidiError> {
        let lower_name = name.to_lowercase();
        for port in ports {
            let port_name = midi_out
                .port_name(port)
                .unwrap_or_else(|_| String::new());
            if port_name.to_lowercase().contains(&lower_name) {
                return Ok((port, port_name));
            }
        }
        Err(MidiError::PortNotFound(name.to_string()))
    }
}

impl Default for MidiEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic protocol handler for unknown MIDI devices.
/// Uses configurable CC assignments from MidiConfig.
struct GenericProtocol;

impl ProtocolHandler for GenericProtocol {
    fn name(&self) -> &str {
        "Generic MIDI"
    }

    fn send_bpm(&self, bpm: f64, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        use super::types::{ControlNumber, MidiValue};
        Ok(vec![MidiMessage::ControlChange(
            config.bpm_channel,
            ControlNumber::new(config.bpm_cc)?,
            MidiValue::from_bpm(bpm),
        )])
    }

    fn send_key(&self, key: &str, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        use super::protocols::key_to_midi_note;
        use super::types::{ControlNumber, MidiValue};
        let note = key_to_midi_note(key);
        Ok(vec![MidiMessage::ControlChange(
            config.key_channel,
            ControlNumber::new(config.key_cc)?,
            MidiValue::new(note)?,
        )])
    }

    fn send_energy(&self, energy: u8, config: &MidiConfig) -> Result<Vec<MidiMessage>, MidiError> {
        use super::types::{ControlNumber, MidiValue};
        Ok(vec![MidiMessage::ControlChange(
            config.energy_channel,
            ControlNumber::new(config.energy_cc)?,
            MidiValue::from_energy(energy),
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_default_not_connected() {
        let engine = MidiEngine::new();
        assert!(!engine.is_connected());
        assert!(engine.connected_device().is_none());
        assert!(engine.protocol_name().is_none());
    }

    #[test]
    fn engine_default_config() {
        let engine = MidiEngine::new();
        let config = engine.config();
        assert_eq!(config.bpm_cc, 1);
        assert_eq!(config.key_cc, 2);
        assert_eq!(config.energy_cc, 3);
    }

    #[test]
    fn engine_custom_config() {
        let mut config = MidiConfig::default();
        config.bpm_cc = 10;
        config.send_clock = true;
        let engine = MidiEngine::with_config(config);
        assert_eq!(engine.config().bpm_cc, 10);
        assert!(engine.config().send_clock);
    }

    #[test]
    fn engine_set_config() {
        let mut engine = MidiEngine::new();
        let mut config = MidiConfig::default();
        config.bpm_cc = 20;
        engine.set_config(config);
        assert_eq!(engine.config().bpm_cc, 20);
    }

    #[test]
    fn send_message_not_connected() {
        let mut engine = MidiEngine::new();
        let msg = MidiMessage::Clock;
        assert!(engine.send_message(&msg).is_err());
    }

    #[test]
    fn send_track_info_not_connected() {
        let mut engine = MidiEngine::new();
        let info = TrackInfo {
            bpm: 128.0,
            key: "Am".to_string(),
            energy: 75,
            title: "Test".to_string(),
            artist: "DJ".to_string(),
        };
        assert!(engine.send_track_info(&info).is_err());
    }

    #[test]
    fn disconnect_when_not_connected() {
        let mut engine = MidiEngine::new();
        assert!(engine.disconnect().is_err());
    }

    #[test]
    fn list_devices_succeeds() {
        // This test passes even with no MIDI hardware connected
        let engine = MidiEngine::new();
        let result = engine.list_devices();
        assert!(result.is_ok());
    }

    #[test]
    fn connect_nonexistent_device() {
        let mut engine = MidiEngine::new();
        let result = engine.connect("NonExistentDevice12345");
        assert!(result.is_err());
    }

    #[test]
    fn generic_protocol_bpm() {
        let protocol = GenericProtocol;
        let config = MidiConfig::default();
        let messages = protocol.send_bpm(128.0, &config).unwrap();
        assert_eq!(messages.len(), 1);
        let bytes = messages[0].to_bytes();
        assert_eq!(bytes[0], 0xB0);
        assert_eq!(bytes[1], config.bpm_cc);
    }

    #[test]
    fn generic_protocol_key() {
        let protocol = GenericProtocol;
        let config = MidiConfig::default();
        let messages = protocol.send_key("Am", &config).unwrap();
        assert_eq!(messages.len(), 1);
        let bytes = messages[0].to_bytes();
        assert_eq!(bytes[1], config.key_cc);
    }

    #[test]
    fn generic_protocol_energy() {
        let protocol = GenericProtocol;
        let config = MidiConfig::default();
        let messages = protocol.send_energy(80, &config).unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn generic_protocol_name() {
        assert_eq!(GenericProtocol.name(), "Generic MIDI");
    }
}
