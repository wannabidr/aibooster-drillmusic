import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { AnalyticsDashboard } from "@presentation/pages/AnalyticsDashboard";
import { useAnalyticsStore } from "@infrastructure/state/useAnalyticsStore";
import type { AnalyticsDashboardDTO } from "@application/dto/AnalyticsDashboardDTO";

// Mock the useAnalytics hook
vi.mock("@presentation/hooks/useAnalytics", () => ({
  useAnalytics: () => {
    const store = useAnalyticsStore();
    return {
      dashboard: store.dashboard,
      timeRange: store.timeRange,
      isLoading: store.isLoading,
      error: store.error,
      fetchAnalytics: vi.fn(),
    };
  },
}));

const mockDashboard: AnalyticsDashboardDTO = {
  energy: {
    energyCurve: [40, 50, 60, 70, 80],
    peaks: [{ index: 4, value: 80 }],
    valleys: [{ index: 0, value: 40 }],
    avgEnergy: 60,
  },
  genres: {
    items: [
      { genre: "Techno", count: 50, percentage: 50 },
      { genre: "House", count: 50, percentage: 50 },
    ],
    total: 100,
  },
  mixing: {
    commonTransitions: [{ from: "8A", to: "7A", count: 10 }],
    preferredBpmRange: { min: 125, max: 130, avg: 128 },
    keyPreferences: [{ key: "8A", count: 15 }],
    avgTransitionQuality: 82,
  },
  timeline: {
    sessions: [
      {
        sessionId: "s1",
        timestamp: "2026-03-01T22:00:00Z",
        trackCount: 20,
        avgEnergy: 70,
        topGenre: "Techno",
        transitionQuality: 80,
      },
    ],
  },
};

describe("AnalyticsDashboard", () => {
  const mockOnBack = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    useAnalyticsStore.setState({
      dashboard: null,
      timeRange: "30d",
      isLoading: false,
      error: null,
    });
  });

  it("should render back button", () => {
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByTestId("back-button")).toBeTruthy();
  });

  it("should call onBack when back button is clicked", () => {
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    fireEvent.click(screen.getByTestId("back-button"));
    expect(mockOnBack).toHaveBeenCalledOnce();
  });

  it("should render time range buttons", () => {
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByTestId("range-7d")).toBeTruthy();
    expect(screen.getByTestId("range-30d")).toBeTruthy();
    expect(screen.getByTestId("range-90d")).toBeTruthy();
    expect(screen.getByTestId("range-all")).toBeTruthy();
  });

  it("should render heading", () => {
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByText("Analytics")).toBeTruthy();
  });

  it("should show loading state", () => {
    useAnalyticsStore.setState({ isLoading: true });
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByTestId("loading")).toBeTruthy();
  });

  it("should show error state", () => {
    useAnalyticsStore.setState({ error: "Something went wrong" });
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByTestId("error")).toBeTruthy();
    expect(screen.getByText("Something went wrong")).toBeTruthy();
  });

  it("should show empty state when no dashboard data", () => {
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByTestId("empty-state")).toBeTruthy();
  });

  it("should render all chart components when dashboard data exists", () => {
    useAnalyticsStore.setState({ dashboard: mockDashboard });
    render(<AnalyticsDashboard onBack={mockOnBack} />);
    expect(screen.getByText("Energy Report")).toBeTruthy();
    expect(screen.getByText("Genre Distribution")).toBeTruthy();
    expect(screen.getByText("Mixing Patterns")).toBeTruthy();
    expect(screen.getByText("Session History")).toBeTruthy();
  });
});
