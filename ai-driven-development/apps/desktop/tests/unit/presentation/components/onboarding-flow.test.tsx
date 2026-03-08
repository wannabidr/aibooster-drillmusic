import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { OnboardingFlow } from "@presentation/components/onboarding/OnboardingFlow";
import { useOnboardingStore } from "@infrastructure/state/useOnboardingStore";

describe("OnboardingFlow", () => {
  beforeEach(() => {
    useOnboardingStore.getState().reset();
  });

  it("should render the welcome screen at step 0", () => {
    render(<OnboardingFlow />);
    expect(screen.getByText("Welcome to AI DJ Assist")).toBeDefined();
  });

  it("should show step indicator", () => {
    render(<OnboardingFlow />);
    expect(screen.getByTestId("step-indicator")).toBeDefined();
  });

  it("should navigate to library import on Get Started click", () => {
    render(<OnboardingFlow />);
    fireEvent.click(screen.getByText("Get Started"));
    expect(screen.getByText("Import Your Library")).toBeDefined();
  });

  it("should show library import sources at step 1", () => {
    useOnboardingStore.setState({ currentStep: 1 });
    render(<OnboardingFlow />);
    expect(screen.getByText("Rekordbox")).toBeDefined();
    expect(screen.getByText("Serato DJ")).toBeDefined();
    expect(screen.getByText("Traktor")).toBeDefined();
  });

  it("should show history import at step 2", () => {
    useOnboardingStore.setState({ currentStep: 2 });
    render(<OnboardingFlow />);
    expect(screen.getByText("Import Mix History")).toBeDefined();
  });

  it("should show community opt-in at step 3", () => {
    useOnboardingStore.setState({ currentStep: 3 });
    render(<OnboardingFlow />);
    expect(screen.getByText("Join the Community")).toBeDefined();
  });

  it("should show tutorial at step 4", () => {
    useOnboardingStore.setState({ currentStep: 4 });
    render(<OnboardingFlow />);
    expect(screen.getByText("Quick Tour")).toBeDefined();
  });

  it("should show ready screen at step 5", () => {
    useOnboardingStore.setState({ currentStep: 5 });
    render(<OnboardingFlow />);
    expect(screen.getByText("You're All Set!")).toBeDefined();
  });

  it("should allow skipping optional steps (history, community)", () => {
    useOnboardingStore.setState({ currentStep: 2 });
    render(<OnboardingFlow />);
    expect(screen.getByText("Skip")).toBeDefined();
  });

  it("should have a back button on steps > 0", () => {
    useOnboardingStore.setState({ currentStep: 2 });
    render(<OnboardingFlow />);
    const backBtn = screen.getByText("Back");
    expect(backBtn).toBeDefined();
    fireEvent.click(backBtn);
    expect(useOnboardingStore.getState().currentStep).toBe(1);
  });

  it("should not have a back button on step 0", () => {
    render(<OnboardingFlow />);
    expect(screen.queryByText("Back")).toBeNull();
  });

  it("should toggle community opt-in", () => {
    useOnboardingStore.setState({ currentStep: 3 });
    render(<OnboardingFlow />);
    const checkbox = screen.getByLabelText("I want to join the community and share anonymous data");
    fireEvent.click(checkbox);
    expect(useOnboardingStore.getState().communityOptIn).toBe(true);
  });

  it("should complete onboarding from ready screen", () => {
    useOnboardingStore.setState({ currentStep: 5 });
    render(<OnboardingFlow />);
    fireEvent.click(screen.getByText("Start DJing"));
    expect(useOnboardingStore.getState().completed).toBe(true);
  });
});
