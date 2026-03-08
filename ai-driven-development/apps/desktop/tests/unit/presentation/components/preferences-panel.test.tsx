import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PreferencesPanel } from "@presentation/components/PreferencesPanel";
import { usePreferencesStore } from "@infrastructure/state/usePreferencesStore";

describe("PreferencesPanel", () => {
  beforeEach(() => {
    usePreferencesStore.getState().reset();
  });

  it("should render weight sliders for all signal dimensions", () => {
    render(<PreferencesPanel />);
    expect(screen.getByLabelText("BPM")).toBeDefined();
    expect(screen.getByLabelText("Key")).toBeDefined();
    expect(screen.getByLabelText("Energy")).toBeDefined();
    expect(screen.getByLabelText("Genre")).toBeDefined();
    expect(screen.getByLabelText("History")).toBeDefined();
  });

  it("should render energy direction selector", () => {
    render(<PreferencesPanel />);
    expect(screen.getByLabelText("Build")).toBeDefined();
    expect(screen.getByLabelText("Maintain")).toBeDefined();
    expect(screen.getByLabelText("Drop")).toBeDefined();
  });

  it("should render blend style selector", () => {
    render(<PreferencesPanel />);
    expect(screen.getByLabelText("Long Blend")).toBeDefined();
    expect(screen.getByLabelText("Short Cut")).toBeDefined();
    expect(screen.getByLabelText("Echo Out")).toBeDefined();
    expect(screen.getByLabelText("Filter Sweep")).toBeDefined();
    expect(screen.getByLabelText("Backspin")).toBeDefined();
  });

  it("should update store when BPM weight slider changes", () => {
    render(<PreferencesPanel />);
    const slider = screen.getByLabelText("BPM") as HTMLInputElement;
    fireEvent.change(slider, { target: { value: "35" } });
    expect(usePreferencesStore.getState().weights.bpm).toBe(35);
  });

  it("should update store when energy direction changes", () => {
    render(<PreferencesPanel />);
    fireEvent.click(screen.getByLabelText("Build"));
    expect(usePreferencesStore.getState().energyDirection).toBe("build");
  });

  it("should update store when blend style changes", () => {
    render(<PreferencesPanel />);
    fireEvent.click(screen.getByLabelText("Echo Out"));
    expect(usePreferencesStore.getState().blendStyle).toBe("echo_out");
  });

  it("should display current weight values", () => {
    render(<PreferencesPanel />);
    const slider = screen.getByLabelText("BPM") as HTMLInputElement;
    expect(slider.value).toBe("20");
  });

  it("should highlight the active energy direction", () => {
    usePreferencesStore.setState({ energyDirection: "drop" });
    render(<PreferencesPanel />);
    const dropBtn = screen.getByLabelText("Drop");
    expect(dropBtn.getAttribute("data-active")).toBe("true");
  });

  it("should highlight the active blend style", () => {
    usePreferencesStore.setState({ blendStyle: "backspin" });
    render(<PreferencesPanel />);
    const btn = screen.getByLabelText("Backspin");
    expect(btn.getAttribute("data-active")).toBe("true");
  });
});
