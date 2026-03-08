import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CamelotWheel } from "@presentation/components/CamelotWheel";

describe("CamelotWheel", () => {
  it("should render all 24 Camelot positions", () => {
    render(<CamelotWheel currentKey="8A" onKeySelect={() => {}} />);
    for (let n = 1; n <= 12; n++) {
      expect(screen.getByText(`${n}A`)).toBeDefined();
      expect(screen.getByText(`${n}B`)).toBeDefined();
    }
  });

  it("should highlight the current key position", () => {
    render(<CamelotWheel currentKey="5B" onKeySelect={() => {}} />);
    const currentEl = screen.getByText("5B").closest("[data-position]");
    expect(currentEl?.getAttribute("data-state")).toBe("current");
  });

  it("should highlight compatible keys (adjacent +/-1, mode switch)", () => {
    render(<CamelotWheel currentKey="8A" onKeySelect={() => {}} />);
    // Adjacent same letter: 7A, 9A
    expect(
      screen.getByText("7A").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("compatible");
    expect(
      screen.getByText("9A").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("compatible");
    // Mode switch: 8B
    expect(
      screen.getByText("8B").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("compatible");
  });

  it("should wrap around: 12A current highlights 1A and 11A as compatible", () => {
    render(<CamelotWheel currentKey="12A" onKeySelect={() => {}} />);
    expect(
      screen.getByText("1A").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("compatible");
    expect(
      screen.getByText("11A").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("compatible");
  });

  it("should call onKeySelect when a position is clicked", () => {
    const handler = vi.fn();
    render(<CamelotWheel currentKey="8A" onKeySelect={handler} />);
    fireEvent.click(screen.getByText("3B"));
    expect(handler).toHaveBeenCalledWith("3B");
  });

  it("should mark non-adjacent keys as default state", () => {
    render(<CamelotWheel currentKey="8A" onKeySelect={() => {}} />);
    // 3B is far from 8A
    expect(
      screen.getByText("3B").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("default");
  });

  it("should support selectedKey filter highlighting", () => {
    render(
      <CamelotWheel currentKey="8A" selectedKey="3B" onKeySelect={() => {}} />,
    );
    expect(
      screen.getByText("3B").closest("[data-position]")?.getAttribute("data-state"),
    ).toBe("selected");
  });

  it("should render without crashing when no currentKey provided", () => {
    render(<CamelotWheel onKeySelect={() => {}} />);
    expect(screen.getByText("1A")).toBeDefined();
  });
});
