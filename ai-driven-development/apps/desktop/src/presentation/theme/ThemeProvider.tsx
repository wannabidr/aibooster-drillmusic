import { type ReactNode, useEffect } from "react";
import { useThemeStore } from "../../infrastructure/state/useThemeStore";
import { standardThemeVars, highContrastThemeVars } from "./themeVars";

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const { highContrast, performanceMode } = useThemeStore();

  useEffect(() => {
    const vars = highContrast ? highContrastThemeVars : standardThemeVars;
    const root = document.documentElement;
    for (const [key, value] of Object.entries(vars)) {
      root.style.setProperty(key, value);
    }
  }, [highContrast]);

  return (
    <div
      data-high-contrast={String(highContrast)}
      data-performance-mode={String(performanceMode)}
      style={{ display: "contents" }}
    >
      {children}
    </div>
  );
}
