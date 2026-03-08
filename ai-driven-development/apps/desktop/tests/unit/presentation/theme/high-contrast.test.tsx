import { describe, it, expect } from "vitest";
import {
  standardThemeVars,
  highContrastThemeVars,
} from "@presentation/theme/themeVars";

describe("High-Contrast Theme", () => {
  it("should have brighter text-primary than standard theme", () => {
    // High-contrast should use pure white or near-white
    expect(highContrastThemeVars["--text-primary"]).toBe("#ffffff");
    expect(standardThemeVars["--text-primary"]).toBe("#e8e8f0");
  });

  it("should have higher contrast text-secondary", () => {
    // High-contrast secondary text should be brighter than standard
    expect(highContrastThemeVars["--text-secondary"]).not.toBe(
      standardThemeVars["--text-secondary"],
    );
  });

  it("should have darker background base for more contrast", () => {
    expect(highContrastThemeVars["--bg-base"]).toBe("#000000");
  });

  it("should have brighter accent colors", () => {
    expect(highContrastThemeVars["--accent-primary"]).not.toBe(
      standardThemeVars["--accent-primary"],
    );
  });

  it("should have stronger borders for visibility", () => {
    expect(highContrastThemeVars["--border-subtle"]).not.toBe(
      standardThemeVars["--border-subtle"],
    );
  });

  it("should maintain all CSS variable keys from standard theme", () => {
    const standardKeys = Object.keys(standardThemeVars);
    const hcKeys = Object.keys(highContrastThemeVars);
    for (const key of standardKeys) {
      expect(hcKeys).toContain(key);
    }
  });
});
