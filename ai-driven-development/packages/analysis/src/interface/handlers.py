"""JSON-RPC method handlers."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

from src.application.use_cases.analyze_track import AnalyzeTrack
from src.application.use_cases.batch_analyze import BatchAnalyze
from src.application.use_cases.generate_analytics import GenerateAnalytics
from src.application.use_cases.record_session import RecordSession
from src.domain.entities.mix_transition import MixTransition
from src.domain.entities.session_analytics import SessionAnalytics, TrackPlayEvent
from src.domain.ports.mix_history_repository import MixHistoryRepository
from src.domain.services.history_scoring import HistoryScoring
from src.interface.server import JsonRpcServer


def register_handlers(
    server: JsonRpcServer,
    analyze_track: AnalyzeTrack,
    batch_analyze: BatchAnalyze,
    mix_history_repo: MixHistoryRepository | None = None,
    history_scorer: HistoryScoring | None = None,
    record_session: RecordSession | None = None,
    generate_analytics: GenerateAnalytics | None = None,
) -> None:
    def handle_analyze(file_path: str, file_hash: str, force: bool = False) -> dict:
        result = analyze_track.execute(file_path=file_path, file_hash=file_hash, force=force)
        return asdict(result)

    def handle_batch_analyze(directory: str) -> dict[str, Any]:
        result = batch_analyze.execute(directory=directory)
        return {
            "succeeded": [asdict(r) for r in result.succeeded],
            "failed": result.failed,
            "skipped": result.skipped,
        }

    def handle_ping() -> str:
        return "pong"

    server.register("analyze", handle_analyze)
    server.register("batch_analyze", handle_batch_analyze)
    server.register("ping", handle_ping)

    if mix_history_repo is not None and history_scorer is not None:

        def handle_import_history(transitions: list[dict[str, str]]) -> dict[str, int]:
            parsed = []
            for t in transitions:
                ts_str = t.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except ValueError:
                    ts = datetime.now()
                parsed.append(
                    MixTransition(
                        track_a_hash=t["track_a_hash"],
                        track_b_hash=t["track_b_hash"],
                        timestamp=ts,
                        source=t.get("source", "unknown"),
                    )
                )
            mix_history_repo.save_transitions(parsed)
            return {"imported": len(parsed)}

        def handle_score_history(
            current_track_hash: str, candidate_hashes: list[str]
        ) -> dict[str, float]:
            return history_scorer.score_batch(current_track_hash, candidate_hashes)

        server.register("import_history", handle_import_history)
        server.register("score_history", handle_score_history)

    if record_session is not None and generate_analytics is not None:

        def handle_record_session(session_data: dict[str, Any]) -> dict[str, str]:
            tracks = [
                TrackPlayEvent(
                    track_id=uuid.UUID(t["track_id"]),
                    played_at=datetime.fromisoformat(t["played_at"]),
                    energy=t["energy"],
                    genre=t["genre"],
                    bpm=t["bpm"],
                    key=t["key"],
                )
                for t in session_data.get("tracks_played", [])
            ]
            session = SessionAnalytics(
                session_id=uuid.UUID(session_data.get("session_id", str(uuid.uuid4()))),
                timestamp=datetime.fromisoformat(
                    session_data.get("timestamp", datetime.now().isoformat())
                ),
                tracks_played=tracks,
                energy_curve=session_data.get("energy_curve", []),
                genre_distribution=session_data.get("genre_distribution", {}),
                bpm_range=tuple(session_data.get("bpm_range", [0.0, 0.0])),
                key_transitions=[
                    tuple(kt) for kt in session_data.get("key_transitions", [])
                ],
                avg_transition_quality=session_data.get("avg_transition_quality", 0.0),
            )
            record_session.execute(session)
            return {"session_id": str(session.session_id)}

        def handle_get_analytics(days: int = 30) -> dict[str, Any]:
            dashboard = generate_analytics.execute(days=days)
            return asdict(dashboard)

        server.register("record_session", handle_record_session)
        server.register("get_analytics", handle_get_analytics)
