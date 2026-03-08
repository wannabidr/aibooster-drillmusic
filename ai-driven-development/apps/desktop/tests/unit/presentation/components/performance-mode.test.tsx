import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { RecommendationPanel } from "@presentation/components/RecommendationPanel";

const mockRecs = [
  {
    trackId: "t1",
    title: "Bass Drop",
    artist: "DJ Test",
    score: 87,
    bpmScore: 90,
    keyScore: 85,
    energyScore: 80,
    genreScore: 75,
    historyScore: 70,
    bpm: 128,
    camelotPosition: "8A",
  },
  {
    trackId: "t2",
    title: "Synth Wave",
    artist: "DJ Test 2",
    score: 72,
    bpmScore: 70,
    keyScore: 65,
    energyScore: 80,
    genreScore: 75,
    historyScore: 60,
    bpm: 130,
    camelotPosition: "9A",
  },
];

describe("RecommendationPanel - Performance Mode", () => {
  it("should hide detailed score breakdown in performance mode", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
        performanceMode
      />,
    );
    // Score badges (BPM, KEY, NRG, GNR, HST) should be hidden
    expect(screen.queryByText("GNR")).toBeNull();
    expect(screen.queryByText("HST")).toBeNull();
  });

  it("should still show essential info (title, BPM, key) in performance mode", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
        performanceMode
        glanceable
      />,
    );
    expect(screen.getByText("Bass Drop")).toBeDefined();
    expect(screen.getByTestId("glanceable-bpm-t1")).toBeDefined();
    expect(screen.getByTestId("glanceable-key-t1")).toBeDefined();
  });

  it("should still show preview button in performance mode", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
        performanceMode
      />,
    );
    const previewButtons = screen.getAllByText("Preview");
    expect(previewButtons.length).toBeGreaterThan(0);
  });

  it("should hide confidence display in performance mode", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
        performanceMode
      />,
    );
    expect(screen.queryByText("85% confidence")).toBeNull();
  });
});
