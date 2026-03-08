import { describe, it, expect } from "vitest";
import * as fs from "fs";
import * as path from "path";

// Project root is 5 levels up from tests/unit/infrastructure/
const PROJECT_ROOT = path.resolve(__dirname, "../../../../..");

const TAURI_CONF_PATH = path.resolve(
  __dirname,
  "../../../src-tauri/tauri.conf.json",
);

describe("tauri.conf.json Windows configuration", () => {
  let config: Record<string, unknown>;

  it("tauri.conf.json exists and is valid JSON", () => {
    expect(fs.existsSync(TAURI_CONF_PATH)).toBe(true);
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    expect(config).toBeDefined();
  });

  it("has bundle configuration", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    expect(config.bundle).toBeDefined();
  });

  it("has external binary (sidecar) configuration", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    const bundle = config.bundle as Record<string, unknown>;
    expect(bundle.externalBin).toBeDefined();
    expect(Array.isArray(bundle.externalBin)).toBe(true);
    expect(bundle.externalBin).toContain("sidecars/ai-dj-analysis");
  });

  it("has Windows installer config (WiX)", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    const bundle = config.bundle as Record<string, unknown>;
    const windows = bundle.windows as Record<string, unknown>;
    expect(windows).toBeDefined();
    expect(windows.wix).toBeDefined();
  });

  it("has Windows installer config (NSIS)", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    const bundle = config.bundle as Record<string, unknown>;
    const windows = bundle.windows as Record<string, unknown>;
    expect(windows.nsis).toBeDefined();
    const nsis = windows.nsis as Record<string, unknown>;
    expect(nsis.installMode).toBe("currentUser");
  });

  it("has macOS minimum system version", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    const bundle = config.bundle as Record<string, unknown>;
    const macos = bundle.macOS as Record<string, unknown>;
    expect(macos).toBeDefined();
    expect(macos.minimumSystemVersion).toBe("10.15");
  });

  it("has correct product name and identifier", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    expect(config.productName).toBe("AI DJ Assist");
    expect(config.identifier).toBe("com.aidj.assist");
  });

  it("bundle targets all platforms", () => {
    const content = fs.readFileSync(TAURI_CONF_PATH, "utf-8");
    config = JSON.parse(content);
    const bundle = config.bundle as Record<string, unknown>;
    expect(bundle.targets).toBe("all");
  });
});

describe("PyInstaller spec file", () => {
  const SPEC_PATH = path.resolve(
    PROJECT_ROOT,
    "packages/analysis/ai-dj-analysis.spec",
  );

  it("spec file exists", () => {
    expect(fs.existsSync(SPEC_PATH)).toBe(true);
  });

  it("spec targets ai-dj-analysis binary name", () => {
    const content = fs.readFileSync(SPEC_PATH, "utf-8");
    expect(content).toContain('name="ai-dj-analysis"');
  });

  it("spec uses console mode for stdio communication", () => {
    const content = fs.readFileSync(SPEC_PATH, "utf-8");
    expect(content).toContain("console=True");
  });

  it("spec excludes heavy training-only deps", () => {
    const content = fs.readFileSync(SPEC_PATH, "utf-8");
    expect(content).toContain('"torch"');
    expect(content).toContain("excludes");
  });
});

describe("Windows sidecar bundling script", () => {
  const SCRIPT_PATH = path.resolve(
    PROJECT_ROOT,
    "scripts/bundle-sidecar-win.ps1",
  );

  it("script exists", () => {
    expect(fs.existsSync(SCRIPT_PATH)).toBe(true);
  });

  it("script references PyInstaller", () => {
    const content = fs.readFileSync(SCRIPT_PATH, "utf-8");
    expect(content).toContain("pyinstaller");
  });

  it("script copies to Tauri sidecar naming convention", () => {
    const content = fs.readFileSync(SCRIPT_PATH, "utf-8");
    expect(content).toContain("x86_64-pc-windows-msvc");
  });
});

describe("macOS sidecar bundling script", () => {
  const SCRIPT_PATH = path.resolve(
    PROJECT_ROOT,
    "scripts/bundle-sidecar.sh",
  );

  it("script exists", () => {
    expect(fs.existsSync(SCRIPT_PATH)).toBe(true);
  });

  it("script references PyInstaller", () => {
    const content = fs.readFileSync(SCRIPT_PATH, "utf-8");
    expect(content).toContain("pyinstaller");
  });

  it("script handles both arm64 and x86_64", () => {
    const content = fs.readFileSync(SCRIPT_PATH, "utf-8");
    expect(content).toContain("aarch64-apple-darwin");
    expect(content).toContain("x86_64-apple-darwin");
  });
});

describe("GitHub Actions Windows workflow", () => {
  const WORKFLOW_PATH = path.resolve(
    PROJECT_ROOT,
    ".github/workflows/build-windows.yml",
  );

  it("workflow file exists", () => {
    expect(fs.existsSync(WORKFLOW_PATH)).toBe(true);
  });

  it("workflow runs on windows-latest", () => {
    const content = fs.readFileSync(WORKFLOW_PATH, "utf-8");
    expect(content).toContain("windows-latest");
  });

  it("workflow builds Rust for Windows target", () => {
    const content = fs.readFileSync(WORKFLOW_PATH, "utf-8");
    expect(content).toContain("x86_64-pc-windows-msvc");
  });

  it("workflow includes TypeScript test job", () => {
    const content = fs.readFileSync(WORKFLOW_PATH, "utf-8");
    expect(content).toContain("typescript-windows");
  });

  it("workflow includes Tauri build job", () => {
    const content = fs.readFileSync(WORKFLOW_PATH, "utf-8");
    expect(content).toContain("tauri-windows-build");
  });

  it("workflow produces MSI artifact", () => {
    const content = fs.readFileSync(WORKFLOW_PATH, "utf-8");
    expect(content).toContain("msi");
  });
});
