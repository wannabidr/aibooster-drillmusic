import { useSubscriptionStore, Feature, SubscriptionTier } from "../../infrastructure/state/useSubscriptionStore";

interface FeatureGateResult {
  allowed: boolean;
  reason: "allowed" | "needs_contribution" | "needs_pro";
  upgradeMessage: string;
}

const UPGRADE_MESSAGES: Record<string, string> = {
  needs_contribution:
    "Share your anonymized mix history to unlock community-powered recommendations. Your data stays private.",
  needs_pro:
    "Upgrade to Pro to unlock AI blend styles, confidence scoring, and more.",
};

/**
 * Hook to check if the current user can access a feature.
 *
 * Returns the access status and a user-facing upgrade message
 * for gated features. Client-side only -- the backend enforces
 * access independently.
 */
export function useFeatureGate(feature: Feature): FeatureGateResult {
  const { tier, hasContributed, canAccess } = useSubscriptionStore();

  if (canAccess(feature)) {
    return { allowed: true, reason: "allowed", upgradeMessage: "" };
  }

  // Determine why access is denied
  if (tier === "free" && !hasContributed) {
    // Could be unlocked by contributing OR upgrading
    // Check if contributing alone would unlock it
    const contributionWouldUnlock =
      feature === "community_scores";

    if (contributionWouldUnlock) {
      return {
        allowed: false,
        reason: "needs_contribution",
        upgradeMessage: UPGRADE_MESSAGES.needs_contribution,
      };
    }
  }

  return {
    allowed: false,
    reason: "needs_pro",
    upgradeMessage: UPGRADE_MESSAGES.needs_pro,
  };
}
