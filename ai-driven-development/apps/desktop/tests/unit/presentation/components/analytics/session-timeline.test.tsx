import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SessionTimeline } from "@presentation/components/analytics/SessionTimeline";
import type { SessionTimelineDTO } from "@application/dto/AnalyticsDashboardDTO";

const mockData: SessionTimelineDTO = {
  sessions: [
    {
      sessionId: "s1",
      timestamp: "2026-03-01T22:00:00Z",
      trackCount: 24,
      avgEnergy: 72,
      topGenre: "Tech House",
      transitionQuality: 85,
    },
    {
      sessionId: "s2",
      timestamp: "2026-03-04T21:00:00Z",
      trackCount: 18,
      avgEnergy: 65,
      topGenre: "Techno",
      transitionQuality: 70,
    },
    {
      sessionId: "s3",
      timestamp: "2026-03-07T23:00:00Z",
      trackCount: 30,
      avgEnergy: 78,
      topGenre: "Melodic Techno",
      transitionQuality: 45,
    },
  ],
};

describe("SessionTimeline", () => {
  it("should render timeline chart SVG", () => {
    const { container } = render(<SessionTimeline data={mockData} />);
    expect(container.querySelector('[data-testid="session-timeline-chart"]')).toBeTruthy();
  });

  it("should render session dots for each session", () => {
    const { container } = render(<SessionTimeline data={mockData} />);
    const dots = container.querySelectorAll('[data-testid="session-dot"]');
    expect(dots.length).toBe(3);
  });

  it("should render session bars for each session", () => {
    const { container } = render(<SessionTimeline data={mockData} />);
    const bars = container.querySelectorAll('[data-testid="session-bar"]');
    expect(bars.length).toBe(3);
  });

  it("should render energy trend line", () => {
    const { container } = render(<SessionTimeline data={mockData} />);
    const line = container.querySelector('[data-testid="session-energy-line"]');
    expect(line).toBeTruthy();
    expect(line?.getAttribute("d")).toBeTruthy();
  });

  it("should render session list", () => {
    const { container } = render(<SessionTimeline data={mockData} />);
    const list = container.querySelector('[data-testid="session-list"]');
    expect(list).toBeTruthy();
  });

  it("should display genre names in session list", () => {
    render(<SessionTimeline data={mockData} />);
    expect(screen.getByText("Tech House")).toBeTruthy();
    expect(screen.getByText("Techno")).toBeTruthy();
    expect(screen.getByText("Melodic Techno")).toBeTruthy();
  });

  it("should display track counts", () => {
    render(<SessionTimeline data={mockData} />);
    expect(screen.getByText("24 tracks")).toBeTruthy();
    expect(screen.getByText("18 tracks")).toBeTruthy();
    expect(screen.getByText("30 tracks")).toBeTruthy();
  });

  it("should display transition quality percentages", () => {
    render(<SessionTimeline data={mockData} />);
    expect(screen.getByText("85%")).toBeTruthy();
    expect(screen.getByText("70%")).toBeTruthy();
    expect(screen.getByText("45%")).toBeTruthy();
  });

  it("should render empty state when data is null", () => {
    render(<SessionTimeline data={null} />);
    expect(screen.getByText("No session history available")).toBeTruthy();
  });

  it("should render empty state when sessions is empty", () => {
    render(<SessionTimeline data={{ sessions: [] }} />);
    expect(screen.getByText("No session history available")).toBeTruthy();
  });
});
