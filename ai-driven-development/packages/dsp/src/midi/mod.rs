//! MIDI output module for DJ hardware communication.
//!
//! Provides device discovery, connection management, and protocol-specific
//! handlers for Pioneer CDJ, Denon SC6000, and Native Instruments controllers.

pub mod engine;
pub mod protocols;
pub mod types;

pub use engine::MidiEngine;
pub use protocols::ProtocolHandler;
pub use types::{
    DeviceType, MidiChannel, MidiConfig, MidiDevice, MidiError, MidiMessage, MidiValue, TrackInfo,
};
