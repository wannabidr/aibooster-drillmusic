import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ThemeProvider } from "@presentation/theme/ThemeProvider";
import { useThemeStore } from "@infrastructure/state/useThemeStore";

function TestChild() {
  return <div data-testid="child">hello</div>;
}

describe("ThemeProvider", () => {
  beforeEach(() => {
    useThemeStore.setState({
      highContrast: false,
      performanceMode: false,
    });
  });

  it("should render children", () => {
    render(
      <ThemeProvider>
        <TestChild />
      </ThemeProvider>,
    );
    expect(screen.getByTestId("child")).toBeDefined();
  });

  it("should apply data-high-contrast attribute when high-contrast is on", () => {
    useThemeStore.setState({ highContrast: true });
    const { container } = render(
      <ThemeProvider>
        <TestChild />
      </ThemeProvider>,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.getAttribute("data-high-contrast")).toBe("true");
  });

  it("should not apply data-high-contrast attribute when off", () => {
    const { container } = render(
      <ThemeProvider>
        <TestChild />
      </ThemeProvider>,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.getAttribute("data-high-contrast")).toBe("false");
  });

  it("should apply data-performance-mode attribute when performance mode is on", () => {
    useThemeStore.setState({ performanceMode: true });
    const { container } = render(
      <ThemeProvider>
        <TestChild />
      </ThemeProvider>,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.getAttribute("data-performance-mode")).toBe("true");
  });

  it("should not apply data-performance-mode attribute when off", () => {
    const { container } = render(
      <ThemeProvider>
        <TestChild />
      </ThemeProvider>,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.getAttribute("data-performance-mode")).toBe("false");
  });
});
