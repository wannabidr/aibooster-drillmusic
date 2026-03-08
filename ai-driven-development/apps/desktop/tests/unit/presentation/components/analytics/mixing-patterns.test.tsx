import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MixingPatterns } from "@presentation/components/analytics/MixingPatterns";
import type { MixingPatternsDTO } from "@application/dto/AnalyticsDashboardDTO";

const mockData: MixingPatternsDTO = {
  commonTransitions: [
    { from: "8A", to: "7A", count: 12 },
    { from: "5B", to: "6B", count: 8 },
    { from: "3A", to: "4A", count: 6 },
  ],
  preferredBpmRange: { min: 122, max: 132, avg: 127 },
  keyPreferences: [
    { key: "8A", count: 18 },
    { key: "5B", count: 14 },
    { key: "3A", count: 10 },
  ],
  avgTransitionQuality: 78,
};

describe("MixingPatterns", () => {
  it("should render BPM range stats", () => {
    render(<MixingPatterns data={mockData} />);
    expect(screen.getByText("122\u2013132")).toBeTruthy();
  });

  it("should render average BPM", () => {
    render(<MixingPatterns data={mockData} />);
    expect(screen.getByText("127")).toBeTruthy();
  });

  it("should render transition quality", () => {
    render(<MixingPatterns data={mockData} />);
    expect(screen.getByText("78%")).toBeTruthy();
  });

  it("should render transition bars", () => {
    const { container } = render(<MixingPatterns data={mockData} />);
    const bars = container.querySelectorAll('[data-testid="transition-bar"]');
    expect(bars.length).toBe(3);
  });

  it("should render key preference bars", () => {
    const { container } = render(<MixingPatterns data={mockData} />);
    const bars = container.querySelectorAll('[data-testid="key-bar"]');
    expect(bars.length).toBe(3);
  });

  it("should render transitions chart SVG", () => {
    const { container } = render(<MixingPatterns data={mockData} />);
    expect(container.querySelector('[data-testid="transitions-chart"]')).toBeTruthy();
  });

  it("should render keys chart SVG", () => {
    const { container } = render(<MixingPatterns data={mockData} />);
    expect(container.querySelector('[data-testid="keys-chart"]')).toBeTruthy();
  });

  it("should render empty state when data is null", () => {
    render(<MixingPatterns data={null} />);
    expect(screen.getByText("No mixing data available")).toBeTruthy();
  });
});
