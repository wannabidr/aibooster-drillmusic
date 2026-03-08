import React from "react";
import { Feature } from "@infrastructure/state/useSubscriptionStore";
import { useFeatureGate } from "../hooks/useFeatureGate";

interface FeatureGateOverlayProps {
  feature: Feature;
  children: React.ReactNode;
  onUpgradeClick?: () => void;
  onContributeClick?: () => void;
}

/**
 * Wraps a feature component and shows an overlay when the user
 * doesn't have access. Displays the appropriate upgrade or
 * contribution message based on the gate reason.
 */
export function FeatureGateOverlay({
  feature,
  children,
  onUpgradeClick,
  onContributeClick,
}: FeatureGateOverlayProps) {
  const { allowed, reason, upgradeMessage } = useFeatureGate(feature);

  if (allowed) {
    return <>{children}</>;
  }

  return (
    <div style={{ position: "relative" }}>
      <div style={{ opacity: 0.3, pointerEvents: "none" }}>{children}</div>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "12px",
          padding: "24px",
          textAlign: "center",
        }}
      >
        <p style={{ fontSize: "14px", lineHeight: "1.5" }}>{upgradeMessage}</p>
        {reason === "needs_contribution" && onContributeClick && (
          <button
            onClick={onContributeClick}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Share Mix History
          </button>
        )}
        {reason === "needs_pro" && onUpgradeClick && (
          <button
            onClick={onUpgradeClick}
            style={{
              padding: "8px 16px",
              borderRadius: "6px",
              border: "none",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Upgrade to Pro
          </button>
        )}
      </div>
    </div>
  );
}
