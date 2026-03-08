import { describe, it, expect, beforeEach } from "vitest";
import { useSubscriptionStore } from "@infrastructure/state/useSubscriptionStore";

describe("useSubscriptionStore", () => {
  beforeEach(() => {
    useSubscriptionStore.getState().reset();
  });

  it("should default to free tier with no contribution", () => {
    const state = useSubscriptionStore.getState();
    expect(state.tier).toBe("free");
    expect(state.hasContributed).toBe(false);
  });

  it("should update subscription tier", () => {
    useSubscriptionStore.getState().setTier("pro");
    expect(useSubscriptionStore.getState().tier).toBe("pro");
  });

  it("should update contribution status", () => {
    useSubscriptionStore.getState().setHasContributed(true);
    expect(useSubscriptionStore.getState().hasContributed).toBe(true);
  });

  it("should reset to defaults", () => {
    const store = useSubscriptionStore.getState();
    store.setTier("pro");
    store.setHasContributed(true);
    store.setStripeCustomerId("cus_123");
    store.reset();

    const state = useSubscriptionStore.getState();
    expect(state.tier).toBe("free");
    expect(state.hasContributed).toBe(false);
    expect(state.stripeCustomerId).toBeNull();
  });

  describe("canAccess - FREE tier (no contribution)", () => {
    it("allows local_analysis", () => {
      expect(useSubscriptionStore.getState().canAccess("local_analysis")).toBe(true);
    });

    it("allows basic_recommendations", () => {
      expect(useSubscriptionStore.getState().canAccess("basic_recommendations")).toBe(true);
    });

    it("allows crossfade_preview", () => {
      expect(useSubscriptionStore.getState().canAccess("crossfade_preview")).toBe(true);
    });

    it("denies community_scores", () => {
      expect(useSubscriptionStore.getState().canAccess("community_scores")).toBe(false);
    });

    it("denies ai_blend_styles", () => {
      expect(useSubscriptionStore.getState().canAccess("ai_blend_styles")).toBe(false);
    });

    it("denies confidence_scoring", () => {
      expect(useSubscriptionStore.getState().canAccess("confidence_scoring")).toBe(false);
    });
  });

  describe("canAccess - FREE tier (with contribution)", () => {
    beforeEach(() => {
      useSubscriptionStore.getState().setHasContributed(true);
    });

    it("allows community_scores", () => {
      expect(useSubscriptionStore.getState().canAccess("community_scores")).toBe(true);
    });

    it("denies ai_blend_styles", () => {
      expect(useSubscriptionStore.getState().canAccess("ai_blend_styles")).toBe(false);
    });

    it("denies confidence_scoring", () => {
      expect(useSubscriptionStore.getState().canAccess("confidence_scoring")).toBe(false);
    });
  });

  describe("canAccess - PRO tier", () => {
    beforeEach(() => {
      useSubscriptionStore.getState().setTier("pro");
    });

    it("allows all features", () => {
      const features = [
        "local_analysis",
        "basic_recommendations",
        "crossfade_preview",
        "community_scores",
        "ai_blend_styles",
        "confidence_scoring",
      ] as const;

      for (const feature of features) {
        expect(useSubscriptionStore.getState().canAccess(feature)).toBe(true);
      }
    });
  });

  describe("canAccess - unknown feature", () => {
    it("denies unknown features", () => {
      useSubscriptionStore.getState().setTier("pro");
      expect(useSubscriptionStore.getState().canAccess("nonexistent" as any)).toBe(false);
    });
  });
});
