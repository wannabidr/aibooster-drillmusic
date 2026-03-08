use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

/// JSON-RPC 2.0 request
#[derive(serde::Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    params: serde_json::Value,
}

/// JSON-RPC 2.0 response
#[derive(serde::Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: u64,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(serde::Deserialize)]
struct JsonRpcError {
    #[allow(dead_code)]
    code: i64,
    message: String,
}

/// Manages the Python analysis sidecar process and JSON-RPC communication.
pub struct SidecarManager {
    process: Mutex<Option<Child>>,
    request_id: Mutex<u64>,
}

impl SidecarManager {
    pub fn new() -> Self {
        Self {
            process: Mutex::new(None),
            request_id: Mutex::new(0),
        }
    }

    /// Start the Python sidecar process.
    pub fn start(&self, python_path: &str) -> Result<(), String> {
        let child = Command::new(python_path)
            .args(["-m", "src.interface.server"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start sidecar: {}", e))?;

        let mut proc = self.process.lock().map_err(|e| e.to_string())?;
        *proc = Some(child);
        Ok(())
    }

    /// Stop the Python sidecar process.
    pub fn stop(&self) -> Result<(), String> {
        let mut proc = self.process.lock().map_err(|e| e.to_string())?;
        if let Some(ref mut child) = *proc {
            child.kill().map_err(|e| format!("Failed to kill sidecar: {}", e))?;
            child.wait().map_err(|e| format!("Failed to wait for sidecar: {}", e))?;
        }
        *proc = None;
        Ok(())
    }

    /// Send a JSON-RPC request and receive a response.
    pub fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let mut id_guard = self.request_id.lock().map_err(|e| e.to_string())?;
        *id_guard += 1;
        let id = *id_guard;
        drop(id_guard);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_string(),
            params,
        };

        let request_str =
            serde_json::to_string(&request).map_err(|e| format!("Serialize error: {}", e))?;

        let mut proc = self.process.lock().map_err(|e| e.to_string())?;
        let child = proc
            .as_mut()
            .ok_or_else(|| "Sidecar not running".to_string())?;

        // Write request to stdin
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| "No stdin".to_string())?;
        writeln!(stdin, "{}", request_str).map_err(|e| format!("Write error: {}", e))?;
        stdin.flush().map_err(|e| format!("Flush error: {}", e))?;

        // Read response from stdout
        let stdout = child
            .stdout
            .as_mut()
            .ok_or_else(|| "No stdout".to_string())?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        reader
            .read_line(&mut line)
            .map_err(|e| format!("Read error: {}", e))?;

        let response: JsonRpcResponse =
            serde_json::from_str(&line).map_err(|e| format!("Parse error: {}", e))?;

        if let Some(error) = response.error {
            return Err(format!("RPC error: {}", error.message));
        }

        response
            .result
            .ok_or_else(|| "Empty result".to_string())
    }

    /// Check if the sidecar is running and responsive.
    pub fn health_check(&self) -> bool {
        self.call("ping", serde_json::json!({}))
            .map(|v| v == serde_json::json!("pong"))
            .unwrap_or(false)
    }
}

impl Drop for SidecarManager {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Resolve the sidecar binary name for the current platform.
pub fn sidecar_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "ai-dj-analysis-x86_64-pc-windows-msvc.exe"
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            "ai-dj-analysis-aarch64-apple-darwin"
        } else {
            "ai-dj-analysis-x86_64-apple-darwin"
        }
    } else {
        "ai-dj-analysis-x86_64-unknown-linux-gnu"
    }
}

/// Resolve the sidecar directory relative to the app resource path.
pub fn sidecar_dir(resource_dir: &std::path::Path) -> std::path::PathBuf {
    resource_dir.join("sidecars")
}

/// Full path to the sidecar binary for this platform.
pub fn sidecar_path(resource_dir: &std::path::Path) -> std::path::PathBuf {
    sidecar_dir(resource_dir).join(sidecar_binary_name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_manager_new_creates_instance() {
        let mgr = SidecarManager::new();
        assert!(!mgr.health_check());
    }

    #[test]
    fn sidecar_manager_call_without_start_returns_error() {
        let mgr = SidecarManager::new();
        let result = mgr.call("ping", serde_json::json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Sidecar not running"));
    }

    #[test]
    fn sidecar_manager_stop_without_start_is_ok() {
        let mgr = SidecarManager::new();
        assert!(mgr.stop().is_ok());
    }

    #[test]
    fn sidecar_manager_start_nonexistent_binary_fails() {
        let mgr = SidecarManager::new();
        let result = mgr.start("/nonexistent/binary/path");
        assert!(result.is_err());
    }

    #[test]
    fn sidecar_binary_name_is_platform_correct() {
        let name = sidecar_binary_name();
        if cfg!(target_os = "windows") {
            assert!(name.ends_with(".exe"));
            assert!(name.contains("windows"));
        } else if cfg!(target_os = "macos") {
            assert!(!name.ends_with(".exe"));
            assert!(name.contains("apple-darwin"));
        } else {
            assert!(name.contains("linux"));
        }
    }

    #[test]
    fn sidecar_path_resolves_correctly() {
        let resource = std::path::Path::new("/app/resources");
        let path = sidecar_path(resource);
        assert!(path.starts_with("/app/resources/sidecars"));
        assert!(path.to_string_lossy().contains("ai-dj-analysis"));
    }

    #[test]
    fn sidecar_dir_appends_sidecars() {
        let resource = std::path::Path::new("/test/dir");
        let dir = sidecar_dir(resource);
        assert_eq!(dir, std::path::PathBuf::from("/test/dir/sidecars"));
    }

    #[test]
    fn json_rpc_request_serializes_correctly() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "test_method".to_string(),
            params: serde_json::json!({"key": "value"}),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"test_method\""));
        assert!(json.contains("\"id\":1"));
    }

    #[test]
    fn json_rpc_response_deserializes_success() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":"pong","error":null}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.result, Some(serde_json::json!("pong")));
        assert!(resp.error.is_none());
    }

    #[test]
    fn json_rpc_response_deserializes_error() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":null,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_none());
        assert_eq!(resp.error.unwrap().message, "Method not found");
    }

    #[test]
    fn request_id_increments() {
        let mgr = SidecarManager::new();
        {
            let mut id = mgr.request_id.lock().unwrap();
            assert_eq!(*id, 0);
            *id += 1;
            assert_eq!(*id, 1);
        }
    }
}
