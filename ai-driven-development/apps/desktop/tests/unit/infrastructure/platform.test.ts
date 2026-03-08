import { describe, it, expect } from "vitest";
import {
  basename,
  normalizePath,
  joinPath,
  pathSeparator,
  detectPlatform,
  isWindows,
} from "../../../src/infrastructure/platform";

describe("platform utilities", () => {
  describe("basename", () => {
    it("extracts filename from unix path", () => {
      expect(basename("/home/dj/music/track.wav")).toBe("track.wav");
    });

    it("extracts filename from windows path", () => {
      expect(basename("C:\\Users\\DJ\\Music\\track.wav")).toBe("track.wav");
    });

    it("handles mixed separators", () => {
      expect(basename("C:\\Users/DJ/Music\\track.wav")).toBe("track.wav");
    });

    it("handles filename only", () => {
      expect(basename("track.wav")).toBe("track.wav");
    });

    it("handles trailing separator", () => {
      expect(basename("/home/dj/music/")).toBe("");
    });

    it("handles empty string", () => {
      expect(basename("")).toBe("");
    });
  });

  describe("detectPlatform", () => {
    it("returns a valid platform type", () => {
      const platform = detectPlatform();
      expect(["windows", "macos", "linux", "unknown"]).toContain(platform);
    });
  });

  describe("normalizePath", () => {
    it("handles forward slashes consistently", () => {
      const result = normalizePath("/home/dj/music/track.wav");
      // On non-windows: stays as-is; on windows: converted to backslash
      if (isWindows()) {
        expect(result).toBe("\\home\\dj\\music\\track.wav");
      } else {
        expect(result).toBe("/home/dj/music/track.wav");
      }
    });

    it("handles backslashes consistently", () => {
      const result = normalizePath("C:\\Users\\DJ\\Music\\track.wav");
      if (isWindows()) {
        expect(result).toBe("C:\\Users\\DJ\\Music\\track.wav");
      } else {
        expect(result).toBe("C:/Users/DJ/Music/track.wav");
      }
    });
  });

  describe("pathSeparator", () => {
    it("returns / or \\", () => {
      const sep = pathSeparator();
      expect(["/", "\\"]).toContain(sep);
    });
  });

  describe("joinPath", () => {
    it("joins path segments with platform separator", () => {
      const sep = pathSeparator();
      const result = joinPath("/home", "dj", "music");
      expect(result).toBe(`/home${sep}dj${sep}music`);
    });

    it("strips trailing separators from segments", () => {
      const sep = pathSeparator();
      const result = joinPath("/home/", "dj/", "music");
      expect(result).toBe(`/home${sep}dj${sep}music`);
    });

    it("handles single segment", () => {
      expect(joinPath("music")).toBe("music");
    });
  });
});
