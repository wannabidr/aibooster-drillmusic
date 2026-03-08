import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { NowPlayingBar } from "@presentation/components/NowPlayingBar";

describe("NowPlayingBar - Glanceable Typography", () => {
  it("should display BPM with glanceable styling when glanceable prop is true", () => {
    render(
      <NowPlayingBar
        title="Test Track"
        artist="Test Artist"
        bpm={128}
        camelotKey="8A"
        energy={75}
        glanceable
      />,
    );
    const bpmEl = screen.getByTestId("glanceable-bpm");
    expect(bpmEl).toBeDefined();
    expect(bpmEl.textContent).toContain("128");
  });

  it("should display key with glanceable styling when glanceable prop is true", () => {
    render(
      <NowPlayingBar
        title="Test Track"
        artist="Test Artist"
        bpm={128}
        camelotKey="8A"
        energy={75}
        glanceable
      />,
    );
    const keyEl = screen.getByTestId("glanceable-key");
    expect(keyEl).toBeDefined();
    expect(keyEl.textContent).toContain("8A");
  });

  it("should not show glanceable elements when glanceable is false", () => {
    render(
      <NowPlayingBar
        title="Test Track"
        artist="Test Artist"
        bpm={128}
        camelotKey="8A"
        energy={75}
      />,
    );
    expect(screen.queryByTestId("glanceable-bpm")).toBeNull();
  });
});
