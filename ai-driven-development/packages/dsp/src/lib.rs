pub mod domain;
pub mod engine;
pub mod blend;
pub mod io;
pub mod ffi;
pub mod midi;
pub mod audio_output;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }
}
