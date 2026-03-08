import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GenreDistribution } from "@presentation/components/analytics/GenreDistribution";
import type { GenreDistributionDTO } from "@application/dto/AnalyticsDashboardDTO";

const mockData: GenreDistributionDTO = {
  items: [
    { genre: "Tech House", count: 45, percentage: 36 },
    { genre: "Techno", count: 30, percentage: 24 },
    { genre: "Deep House", count: 25, percentage: 20 },
    { genre: "Melodic Techno", count: 15, percentage: 12 },
    { genre: "Minimal", count: 10, percentage: 8 },
  ],
  total: 125,
};

describe("GenreDistribution", () => {
  it("should render donut chart SVG", () => {
    const { container } = render(<GenreDistribution data={mockData} />);
    expect(container.querySelector('[data-testid="genre-chart"]')).toBeTruthy();
  });

  it("should render correct number of genre slices", () => {
    const { container } = render(<GenreDistribution data={mockData} />);
    const slices = container.querySelectorAll('[data-testid="genre-slice"]');
    expect(slices.length).toBe(5);
  });

  it("should display total track count in center", () => {
    render(<GenreDistribution data={mockData} />);
    expect(screen.getByText("125")).toBeTruthy();
  });

  it("should display genre labels in legend", () => {
    render(<GenreDistribution data={mockData} />);
    expect(screen.getByText("Tech House")).toBeTruthy();
    expect(screen.getByText("Techno")).toBeTruthy();
    expect(screen.getByText("Deep House")).toBeTruthy();
  });

  it("should display percentages in legend", () => {
    render(<GenreDistribution data={mockData} />);
    expect(screen.getByText("36%")).toBeTruthy();
    expect(screen.getByText("24%")).toBeTruthy();
  });

  it("should render empty state when data is null", () => {
    render(<GenreDistribution data={null} />);
    expect(screen.getByText("No genre data available")).toBeTruthy();
  });

  it("should render empty state when items is empty", () => {
    render(<GenreDistribution data={{ items: [], total: 0 }} />);
    expect(screen.getByText("No genre data available")).toBeTruthy();
  });
});
