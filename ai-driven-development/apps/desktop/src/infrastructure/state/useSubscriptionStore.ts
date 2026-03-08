import { create } from "zustand";

export type SubscriptionTier = "free" | "pro";

export type Feature =
  | "local_analysis"
  | "basic_recommendations"
  | "crossfade_preview"
  | "community_scores"
  | "ai_blend_styles"
  | "confidence_scoring";

/**
 * Feature access matrix matching the backend's SubscriptionGate.
 *
 * Client-side checks are for UX only (show/hide, enable/disable).
 * The backend enforces access server-side.
 */
const FEATURE_ACCESS: Record<Feature, { free: boolean; freeContributed: boolean; pro: boolean }> = {
  local_analysis:        { free: true,  freeContributed: true,  pro: true },
  basic_recommendations: { free: true,  freeContributed: true,  pro: true },
  crossfade_preview:     { free: true,  freeContributed: true,  pro: true },
  community_scores:      { free: false, freeContributed: true,  pro: true },
  ai_blend_styles:       { free: false, freeContributed: false, pro: true },
  confidence_scoring:    { free: false, freeContributed: false, pro: true },
};

interface SubscriptionState {
  tier: SubscriptionTier;
  hasContributed: boolean;
  isTrialing: boolean;
  trialEndsAt: string | null;
  stripeCustomerId: string | null;

  setTier: (tier: SubscriptionTier) => void;
  setHasContributed: (contributed: boolean) => void;
  setTrialInfo: (isTrialing: boolean, trialEndsAt: string | null) => void;
  setStripeCustomerId: (id: string | null) => void;
  canAccess: (feature: Feature) => boolean;
  reset: () => void;
}

export const useSubscriptionStore = create<SubscriptionState>((set, get) => ({
  tier: "free",
  hasContributed: false,
  isTrialing: false,
  trialEndsAt: null,
  stripeCustomerId: null,

  setTier: (tier) => set({ tier }),
  setHasContributed: (contributed) => set({ hasContributed: contributed }),
  setTrialInfo: (isTrialing, trialEndsAt) => set({ isTrialing, trialEndsAt }),
  setStripeCustomerId: (id) => set({ stripeCustomerId: id }),

  canAccess: (feature: Feature): boolean => {
    const state = get();
    const matrix = FEATURE_ACCESS[feature];
    if (!matrix) return false;

    if (state.tier === "pro") return matrix.pro;
    if (state.hasContributed) return matrix.freeContributed;
    return matrix.free;
  },

  reset: () =>
    set({
      tier: "free",
      hasContributed: false,
      isTrialing: false,
      trialEndsAt: null,
      stripeCustomerId: null,
    }),
}));
