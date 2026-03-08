import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { EnergyGraph } from "@presentation/components/EnergyGraph";

describe("EnergyGraph", () => {
  it("should render SVG element", () => {
    const { container } = render(
      <EnergyGraph currentEnergyCurve={[50, 55, 60, 65, 70]} />,
    );
    expect(container.querySelector("svg")).toBeDefined();
  });

  it("should render current track energy path", () => {
    const { container } = render(
      <EnergyGraph currentEnergyCurve={[30, 40, 50, 60, 70]} />,
    );
    const paths = container.querySelectorAll("path[data-role='current']");
    expect(paths.length).toBe(1);
  });

  it("should render overlay paths for recommended tracks", () => {
    const { container } = render(
      <EnergyGraph
        currentEnergyCurve={[50, 55, 60]}
        overlays={[
          { trackId: "a", curve: [45, 50, 55], label: "Track A" },
          { trackId: "b", curve: [60, 65, 70], label: "Track B" },
        ]}
      />,
    );
    const overlays = container.querySelectorAll("path[data-role='overlay']");
    expect(overlays.length).toBe(2);
  });

  it("should display energy direction indicator", () => {
    render(
      <EnergyGraph
        currentEnergyCurve={[30, 40, 50, 60, 70]}
        direction="build"
      />,
    );
    expect(screen.getByText("Build")).toBeDefined();
  });

  it("should show 'Maintain' direction for flat curves", () => {
    render(
      <EnergyGraph
        currentEnergyCurve={[50, 50, 50, 50]}
        direction="maintain"
      />,
    );
    expect(screen.getByText("Maintain")).toBeDefined();
  });

  it("should show 'Drop' direction for decreasing curves", () => {
    render(
      <EnergyGraph
        currentEnergyCurve={[80, 70, 60, 50]}
        direction="drop"
      />,
    );
    expect(screen.getByText("Drop")).toBeDefined();
  });

  it("should render empty state when no curve provided", () => {
    render(<EnergyGraph />);
    expect(screen.getByText("No energy data")).toBeDefined();
  });

  it("should set viewBox for responsive sizing", () => {
    const { container } = render(
      <EnergyGraph currentEnergyCurve={[50, 60, 70]} />,
    );
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("viewBox")).toBeTruthy();
  });
});
