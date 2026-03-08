//! Cross-platform audio output using the `cpal` crate.
//!
//! Abstracts CoreAudio (macOS) and WASAPI (Windows) behind a unified interface.
//! Supports device enumeration, selection, and playback of AudioBuffer data.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};

use super::types::{AudioError, AudioOutputDevice, AudioOutputPort, AudioPlaybackState};
use crate::domain::AudioBuffer;

/// Audio output implementation using cpal.
pub struct CpalAudioOutput {
    host: cpal::Host,
    selected_device: Option<AudioOutputDevice>,
    state: AudioPlaybackState,
    /// Active playback stream (kept alive to sustain playback).
    _stream: Option<Stream>,
}

impl CpalAudioOutput {
    /// Create a new cpal audio output using the default host.
    pub fn new() -> Self {
        Self {
            host: cpal::default_host(),
            selected_device: None,
            state: AudioPlaybackState::Stopped,
            _stream: None,
        }
    }

    /// Find a cpal Device by device name.
    fn find_device(&self, device_id: &str) -> Result<Device, AudioError> {
        let devices = self
            .host
            .output_devices()
            .map_err(|e| AudioError::PlatformError(e.to_string()))?;

        for device in devices {
            let name = device
                .name()
                .unwrap_or_else(|_| "Unknown".to_string());
            if name == device_id {
                return Ok(device);
            }
        }
        Err(AudioError::DeviceNotFound(device_id.to_string()))
    }

    /// Convert a cpal Device to our AudioOutputDevice DTO.
    fn device_to_dto(device: &Device, is_default: bool) -> AudioOutputDevice {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        let config = device
            .default_output_config()
            .map(|c| (c.channels(), c.sample_rate().0))
            .unwrap_or((2, 44100));

        AudioOutputDevice {
            id: name.clone(),
            name,
            channels: config.0,
            sample_rate: config.1,
            is_default,
        }
    }

    /// Build a playback stream for the given device and audio data.
    fn build_stream(
        device: &Device,
        buffer: &AudioBuffer,
    ) -> Result<Stream, AudioError> {
        let config = device
            .default_output_config()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        let stream_config = StreamConfig {
            channels: config.channels(),
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        // Resample if needed: simple nearest-neighbor for now
        let samples = if buffer.sample_rate() != config.sample_rate().0 {
            resample_nearest(
                buffer.samples(),
                buffer.sample_rate(),
                config.sample_rate().0,
                buffer.channels(),
            )
        } else {
            buffer.samples().to_vec()
        };

        // Remix channels if needed
        let output_channels = config.channels() as usize;
        let input_channels = buffer.channels() as usize;
        let samples = if input_channels != output_channels {
            remix_channels(&samples, input_channels, output_channels)
        } else {
            samples
        };

        let data = Arc::new(samples);
        let position = Arc::new(Mutex::new(0usize));

        let data_clone = Arc::clone(&data);
        let pos_clone = Arc::clone(&position);

        let stream = device
            .build_output_stream(
                &stream_config,
                move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut pos = pos_clone.lock().unwrap();
                    for sample in output.iter_mut() {
                        if *pos < data_clone.len() {
                            *sample = data_clone[*pos];
                            *pos += 1;
                        } else {
                            *sample = 0.0; // Silence after buffer ends
                        }
                    }
                },
                move |err| {
                    eprintln!("Audio output error: {err}");
                },
                None,
            )
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        Ok(stream)
    }
}

impl Default for CpalAudioOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioOutputPort for CpalAudioOutput {
    fn list_outputs(&self) -> Result<Vec<AudioOutputDevice>, AudioError> {
        let default_name = self
            .host
            .default_output_device()
            .and_then(|d| d.name().ok());

        let devices = self
            .host
            .output_devices()
            .map_err(|e| AudioError::PlatformError(e.to_string()))?;

        let mut result = Vec::new();
        for device in devices {
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            let is_default = default_name.as_deref() == Some(&name);
            result.push(Self::device_to_dto(&device, is_default));
        }

        if result.is_empty() {
            return Err(AudioError::NoDevices);
        }

        Ok(result)
    }

    fn current_output(&self) -> Option<&AudioOutputDevice> {
        self.selected_device.as_ref()
    }

    fn set_output(&mut self, device_id: &str) -> Result<(), AudioError> {
        // Verify device exists
        let device = self.find_device(device_id)?;
        let dto = Self::device_to_dto(&device, false);

        // Stop any current playback
        if self.state == AudioPlaybackState::Playing {
            self.stop()?;
        }

        self.selected_device = Some(dto);
        Ok(())
    }

    fn play(&mut self, buffer: &AudioBuffer) -> Result<(), AudioError> {
        let device_id = self
            .selected_device
            .as_ref()
            .map(|d| d.id.clone())
            .or_else(|| {
                self.host
                    .default_output_device()
                    .and_then(|d| d.name().ok())
            })
            .ok_or(AudioError::NoOutputSelected)?;

        let device = self.find_device(&device_id)?;

        // Update selected device if using default
        if self.selected_device.is_none() {
            self.selected_device = Some(Self::device_to_dto(&device, true));
        }

        let stream = Self::build_stream(&device, buffer)?;

        stream
            .play()
            .map_err(|e| AudioError::PlaybackError(e.to_string()))?;

        self._stream = Some(stream);
        self.state = AudioPlaybackState::Playing;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), AudioError> {
        if self.state == AudioPlaybackState::Stopped {
            return Err(AudioError::NotPlaying);
        }

        // Dropping the stream stops playback
        self._stream = None;
        self.state = AudioPlaybackState::Stopped;
        Ok(())
    }

    fn state(&self) -> AudioPlaybackState {
        self.state
    }
}

/// Simple nearest-neighbor resampling.
fn resample_nearest(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
    channels: u16,
) -> Vec<f32> {
    let channels = channels as usize;
    let num_frames = samples.len() / channels;
    let ratio = from_rate as f64 / to_rate as f64;
    let new_frames = (num_frames as f64 / ratio).ceil() as usize;

    let mut output = Vec::with_capacity(new_frames * channels);
    for i in 0..new_frames {
        let src_frame = ((i as f64) * ratio) as usize;
        let src_frame = src_frame.min(num_frames - 1);
        for ch in 0..channels {
            output.push(samples[src_frame * channels + ch]);
        }
    }
    output
}

/// Remix audio between different channel counts.
fn remix_channels(samples: &[f32], from_ch: usize, to_ch: usize) -> Vec<f32> {
    if from_ch == 0 || to_ch == 0 {
        return Vec::new();
    }

    let num_frames = samples.len() / from_ch;
    let mut output = Vec::with_capacity(num_frames * to_ch);

    for frame in 0..num_frames {
        let src_start = frame * from_ch;
        if from_ch == 1 && to_ch == 2 {
            // Mono to stereo: duplicate
            let s = samples[src_start];
            output.push(s);
            output.push(s);
        } else if from_ch == 2 && to_ch == 1 {
            // Stereo to mono: average
            let avg = (samples[src_start] + samples[src_start + 1]) / 2.0;
            output.push(avg);
        } else {
            // General case: copy what we can, zero-fill the rest
            for ch in 0..to_ch {
                if ch < from_ch {
                    output.push(samples[src_start + ch]);
                } else {
                    output.push(0.0);
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_same_rate() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let result = resample_nearest(&samples, 44100, 44100, 1);
        assert_eq!(result.len(), 4);
        for (a, b) in samples.iter().zip(result.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn resample_upsample_2x() {
        let samples = vec![0.0, 1.0];
        let result = resample_nearest(&samples, 22050, 44100, 1);
        assert_eq!(result.len(), 4);
        // Nearest-neighbor: each sample repeated
        assert!((result[0] - 0.0).abs() < f32::EPSILON);
        assert!((result[1] - 0.0).abs() < f32::EPSILON);
        assert!((result[2] - 1.0).abs() < f32::EPSILON);
        assert!((result[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn resample_downsample_2x() {
        let samples = vec![0.0, 0.5, 1.0, 0.5];
        let result = resample_nearest(&samples, 44100, 22050, 1);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn resample_stereo() {
        let samples = vec![0.1, 0.2, 0.3, 0.4]; // 2 frames, 2 channels
        let result = resample_nearest(&samples, 44100, 44100, 2);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn remix_mono_to_stereo() {
        let mono = vec![0.5, 1.0, -0.5];
        let stereo = remix_channels(&mono, 1, 2);
        assert_eq!(stereo.len(), 6);
        assert!((stereo[0] - 0.5).abs() < f32::EPSILON);
        assert!((stereo[1] - 0.5).abs() < f32::EPSILON);
        assert!((stereo[2] - 1.0).abs() < f32::EPSILON);
        assert!((stereo[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn remix_stereo_to_mono() {
        let stereo = vec![0.4, 0.6, 1.0, 0.0];
        let mono = remix_channels(&stereo, 2, 1);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < f32::EPSILON); // (0.4+0.6)/2
        assert!((mono[1] - 0.5).abs() < f32::EPSILON); // (1.0+0.0)/2
    }

    #[test]
    fn remix_same_channels() {
        let data = vec![0.1, 0.2, 0.3, 0.4];
        let result = remix_channels(&data, 2, 2);
        assert_eq!(result, data);
    }

    #[test]
    fn remix_empty() {
        let result = remix_channels(&[], 0, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn remix_wider_output() {
        // 1 channel to 4 channels: copy + zero-fill
        let mono = vec![0.5, 1.0];
        let quad = remix_channels(&mono, 1, 4);
        assert_eq!(quad.len(), 8);
        assert!((quad[0] - 0.5).abs() < f32::EPSILON);
        assert!((quad[1] - 0.0).abs() < f32::EPSILON);
        assert!((quad[2] - 0.0).abs() < f32::EPSILON);
        assert!((quad[3] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn cpal_output_initial_state() {
        let output = CpalAudioOutput::new();
        assert_eq!(output.state(), AudioPlaybackState::Stopped);
        assert!(output.current_output().is_none());
    }

    #[test]
    fn cpal_output_list_devices() {
        // Should succeed even if no audio hardware is connected
        let output = CpalAudioOutput::new();
        let result = output.list_outputs();
        // On CI without audio, this may error with NoDevices -- both outcomes are valid
        match result {
            Ok(devices) => {
                assert!(!devices.is_empty());
                // At least one device should be marked as default
                let has_default = devices.iter().any(|d| d.is_default);
                assert!(has_default);
            }
            Err(AudioError::NoDevices) => {
                // Acceptable on headless CI
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn cpal_output_set_nonexistent_device() {
        let mut output = CpalAudioOutput::new();
        let result = output.set_output("NonExistentDevice12345");
        assert!(result.is_err());
    }

    #[test]
    fn cpal_output_stop_when_stopped() {
        let mut output = CpalAudioOutput::new();
        let result = output.stop();
        assert!(result.is_err());
    }

    #[test]
    fn cpal_output_play_without_device_uses_default() {
        let mut output = CpalAudioOutput::new();
        let buffer = AudioBuffer::new(vec![0.0; 4410], 44100, 1).unwrap();
        let result = output.play(&buffer);
        // May succeed (if audio hardware exists) or fail (headless CI)
        match result {
            Ok(()) => {
                assert_eq!(output.state(), AudioPlaybackState::Playing);
                output.stop().unwrap();
            }
            Err(AudioError::NoOutputSelected) | Err(AudioError::NoDevices) => {
                // Acceptable on headless CI
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
