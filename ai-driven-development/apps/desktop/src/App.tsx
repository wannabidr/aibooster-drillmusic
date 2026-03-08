import { useState } from "react";
import "./presentation/theme.css";
import { MainPage } from "./presentation/pages/MainPage";
import { AnalyticsDashboard } from "./presentation/pages/AnalyticsDashboard";
import { ThemeProvider } from "./presentation/theme/ThemeProvider";
import { OnboardingFlow } from "./presentation/components/onboarding/OnboardingFlow";
import { useOnboardingStore } from "./infrastructure/state/useOnboardingStore";

type AppPage = "main" | "analytics";

export function App() {
  const { completed } = useOnboardingStore();
  const [page, setPage] = useState<AppPage>("main");

  if (!completed) {
    return (
      <ThemeProvider>
        <OnboardingFlow />
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider>
      {page === "main" ? (
        <MainPage onNavigateAnalytics={() => setPage("analytics")} />
      ) : (
        <AnalyticsDashboard onBack={() => setPage("main")} />
      )}
    </ThemeProvider>
  );
}
