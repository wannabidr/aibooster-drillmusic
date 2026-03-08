import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { EnergyReport } from "@presentation/components/analytics/EnergyReport";
import type { EnergyReportDTO } from "@application/dto/AnalyticsDashboardDTO";

const mockData: EnergyReportDTO = {
  energyCurve: [30, 45, 60, 80, 70, 55, 40, 65, 85, 60],
  peaks: [
    { index: 3, value: 80 },
    { index: 8, value: 85 },
  ],
  valleys: [
    { index: 0, value: 30 },
    { index: 6, value: 40 },
  ],
  avgEnergy: 59,
};

describe("EnergyReport", () => {
  it("should render SVG chart when data is provided", () => {
    const { container } = render(<EnergyReport data={mockData} />);
    expect(container.querySelector('[data-testid="energy-report-chart"]')).toBeTruthy();
  });

  it("should render the energy curve path", () => {
    const { container } = render(<EnergyReport data={mockData} />);
    const curve = container.querySelector('[data-testid="energy-curve"]');
    expect(curve).toBeTruthy();
    expect(curve?.getAttribute("d")).toBeTruthy();
  });

  it("should render peak markers", () => {
    const { container } = render(<EnergyReport data={mockData} />);
    const peaks = container.querySelectorAll('[data-testid="peak-marker"]');
    expect(peaks.length).toBe(2);
  });

  it("should render valley markers", () => {
    const { container } = render(<EnergyReport data={mockData} />);
    const valleys = container.querySelectorAll('[data-testid="valley-marker"]');
    expect(valleys.length).toBe(2);
  });

  it("should render average line", () => {
    const { container } = render(<EnergyReport data={mockData} />);
    const avgLine = container.querySelector('[data-testid="avg-line"]');
    expect(avgLine).toBeTruthy();
  });

  it("should display average energy stat", () => {
    render(<EnergyReport data={mockData} />);
    expect(screen.getByText("59")).toBeTruthy();
  });

  it("should display peak and valley counts", () => {
    render(<EnergyReport data={mockData} />);
    const allTwos = screen.getAllByText("2");
    // Both peaks and valleys have count 2
    expect(allTwos.length).toBe(2);
  });

  it("should render empty state when data is null", () => {
    render(<EnergyReport data={null} />);
    expect(screen.getByText("No energy data available")).toBeTruthy();
  });

  it("should render empty state when energy curve is empty", () => {
    render(
      <EnergyReport data={{ energyCurve: [], peaks: [], valleys: [], avgEnergy: 0 }} />,
    );
    expect(screen.getByText("No energy data available")).toBeTruthy();
  });
});
