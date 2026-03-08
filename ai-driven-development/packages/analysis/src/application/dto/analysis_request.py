"""Analysis request DTO."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisRequest:
    file_path: str
    force_reanalyze: bool = False
