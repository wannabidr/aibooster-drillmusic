"""Energy profile value object."""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_TRAJECTORIES = {"build", "maintain", "drop"}


@dataclass(frozen=True)
class EnergyProfile:
    overall: float
    segments: list[dict[str, float]] = field(default_factory=list)
    trajectory: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall <= 100.0:
            raise ValueError(f"Energy must be between 0 and 100, got {self.overall}")
        if self.trajectory is not None and self.trajectory not in _VALID_TRAJECTORIES:
            raise ValueError(
                f"Trajectory must be one of {_VALID_TRAJECTORIES}, got {self.trajectory}"
            )
