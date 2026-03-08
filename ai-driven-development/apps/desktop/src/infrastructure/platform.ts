/**
 * Platform detection utilities for cross-platform support.
 * Used to adapt file paths and behaviors for Windows vs macOS.
 */

export type Platform = "windows" | "macos" | "linux" | "unknown";

export function detectPlatform(): Platform {
  if (typeof navigator === "undefined") return "unknown";

  const platform = (navigator.platform || "").toLowerCase();
  if (platform.startsWith("win")) return "windows";
  if (platform.startsWith("mac")) return "macos";
  if (platform.includes("linux")) return "linux";

  // Fallback to userAgent with stricter matching
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes("windows")) return "windows";
  if (ua.includes("macintosh") || ua.includes("mac os")) return "macos";
  if (ua.includes("linux") && !ua.includes("android")) return "linux";
  return "unknown";
}

export function isWindows(): boolean {
  return detectPlatform() === "windows";
}

export function isMacOS(): boolean {
  return detectPlatform() === "macos";
}

/**
 * Normalize a file path for the current platform.
 * Converts forward/back slashes to the platform's preferred separator.
 */
export function normalizePath(path: string): string {
  if (isWindows()) {
    return path.replace(/\//g, "\\");
  }
  return path.replace(/\\/g, "/");
}

/**
 * Get the path separator for the current platform.
 */
export function pathSeparator(): string {
  return isWindows() ? "\\" : "/";
}

/**
 * Join path segments using the platform's separator.
 */
export function joinPath(...segments: string[]): string {
  const sep = pathSeparator();
  return segments
    .map((s) => s.replace(/[/\\]+$/, ""))
    .join(sep);
}

/**
 * Extract filename from a full path, handling both separators.
 */
export function basename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || "";
}
