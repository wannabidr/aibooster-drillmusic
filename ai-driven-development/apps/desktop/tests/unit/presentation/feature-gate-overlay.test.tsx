import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { useSubscriptionStore } from "@infrastructure/state/useSubscriptionStore";
import { FeatureGateOverlay } from "@presentation/components/FeatureGateOverlay";

describe("FeatureGateOverlay", () => {
  beforeEach(() => {
    useSubscriptionStore.getState().reset();
  });

  it("renders children when feature is allowed", () => {
    render(
      <FeatureGateOverlay feature="local_analysis">
        <span>Content</span>
      </FeatureGateOverlay>,
    );
    expect(screen.getByText("Content")).toBeDefined();
  });

  it("shows contribution message for community_scores when not contributed", () => {
    render(
      <FeatureGateOverlay feature="community_scores">
        <span>Scores</span>
      </FeatureGateOverlay>,
    );
    expect(
      screen.getByText(/Share your anonymized mix history/),
    ).toBeDefined();
  });

  it("shows upgrade message for pro-only features", () => {
    render(
      <FeatureGateOverlay feature="ai_blend_styles">
        <span>Blend</span>
      </FeatureGateOverlay>,
    );
    expect(screen.getByText(/Upgrade to Pro/)).toBeDefined();
  });

  it("calls onContributeClick when share button is clicked", () => {
    const onContribute = vi.fn();
    render(
      <FeatureGateOverlay
        feature="community_scores"
        onContributeClick={onContribute}
      >
        <span>Scores</span>
      </FeatureGateOverlay>,
    );
    fireEvent.click(screen.getByText("Share Mix History"));
    expect(onContribute).toHaveBeenCalledOnce();
  });

  it("calls onUpgradeClick when upgrade button is clicked", () => {
    const onUpgrade = vi.fn();
    render(
      <FeatureGateOverlay feature="ai_blend_styles" onUpgradeClick={onUpgrade}>
        <span>Blend</span>
      </FeatureGateOverlay>,
    );
    fireEvent.click(screen.getByText("Upgrade to Pro"));
    expect(onUpgrade).toHaveBeenCalledOnce();
  });

  it("allows community_scores when user has contributed", () => {
    useSubscriptionStore.getState().setHasContributed(true);
    render(
      <FeatureGateOverlay feature="community_scores">
        <span>Scores Available</span>
      </FeatureGateOverlay>,
    );
    const el = screen.getByText("Scores Available");
    // Should be rendered without opacity overlay
    expect(el.closest("div")?.style.opacity).not.toBe("0.3");
  });

  it("allows all features for pro users", () => {
    useSubscriptionStore.getState().setTier("pro");
    render(
      <FeatureGateOverlay feature="ai_blend_styles">
        <span>Pro Content</span>
      </FeatureGateOverlay>,
    );
    const el = screen.getByText("Pro Content");
    expect(el.closest("div")?.style.opacity).not.toBe("0.3");
  });
});
