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
];

describe("RecommendationPanel - Glanceable Typography", () => {
  it("should display BPM value prominently with data-glanceable attribute", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
        glanceable
      />,
    );
    const bpmElement = screen.getByTestId("glanceable-bpm-t1");
    expect(bpmElement).toBeDefined();
    expect(bpmElement.textContent).toContain("128");
  });

  it("should display Camelot key prominently with data-glanceable attribute", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
        glanceable
      />,
    );
    const keyElement = screen.getByTestId("glanceable-key-t1");
    expect(keyElement).toBeDefined();
    expect(keyElement.textContent).toContain("8A");
  });

  it("should not show glanceable elements when glanceable is false", () => {
    render(
      <RecommendationPanel
        recommendations={mockRecs}
        confidence={85}
        isLoading={false}
        onSelect={() => {}}
        onPreview={() => {}}
      />,
    );
    expect(screen.queryByTestId("glanceable-bpm-t1")).toBeNull();
  });
});
