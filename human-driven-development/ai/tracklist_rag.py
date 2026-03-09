import argparse
import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TransitionDoc:
    doc_id: int
    tracklist_url: str
    tracklist_title: str
    track_order: int
    prev_track_2: Optional[str]
    prev_track_1: Optional[str]
    current_track: str
    next_track: str
    query_text: str


def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = " ".join(s.split())
    return s


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_tracks_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"tracklist_url", "tracklist_title", "track_order", "track_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tracks.csv에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df = df.copy()
    df["track_order"] = pd.to_numeric(df["track_order"], errors="coerce")
    df = df.dropna(subset=["tracklist_url", "track_order", "track_title"])
    df["track_order"] = df["track_order"].astype(int)
    df["track_title"] = df["track_title"].astype(str).str.strip()
    df["tracklist_title"] = df["tracklist_title"].fillna("").astype(str).str.strip()

    df = df.sort_values(["tracklist_url", "track_order"]).reset_index(drop=True)
    return df


def build_transition_docs(df: pd.DataFrame) -> List[TransitionDoc]:
    docs: List[TransitionDoc] = []
    doc_id = 0

    for tracklist_url, g in df.groupby("tracklist_url", sort=False):
        g = g.sort_values("track_order").reset_index(drop=True)
        titles = g["track_title"].tolist()
        tl_title = safe_str(g["tracklist_title"].iloc[0])

        for i in range(len(titles) - 1):
            prev2 = titles[i - 2] if i - 2 >= 0 else ""
            prev1 = titles[i - 1] if i - 1 >= 0 else ""
            current = titles[i]
            nxt = titles[i + 1]

            query_text = (
                f"tracklist {tl_title} "
                f"prev2 {safe_str(prev2)} "
                f"prev1 {safe_str(prev1)} "
                f"current {safe_str(current)}"
            )

            docs.append(
                TransitionDoc(
                    doc_id=doc_id,
                    tracklist_url=tracklist_url,
                    tracklist_title=tl_title,
                    track_order=int(g["track_order"].iloc[i]),
                    prev_track_2=safe_str(prev2) or None,
                    prev_track_1=safe_str(prev1) or None,
                    current_track=safe_str(current),
                    next_track=safe_str(nxt),
                    query_text=query_text,
                )
            )
            doc_id += 1

    return docs


def build_index(tracks_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = load_tracks_csv(tracks_csv)
    docs = build_transition_docs(df)

    if not docs:
        raise ValueError("transition 문서가 0개입니다. tracks.csv 내용을 확인하세요.")

    corpus = [d.query_text for d in docs]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform(corpus)

    with open(os.path.join(out_dir, "docs.pkl"), "wb") as f:
        pickle.dump([asdict(d) for d in docs], f)

    with open(os.path.join(out_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(out_dir, "matrix.pkl"), "wb") as f:
        pickle.dump(matrix, f)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tracks_csv": tracks_csv,
                "num_docs": len(docs),
                "index_type": "tfidf_transition_rag",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved index -> {out_dir}")
    print(f"Documents: {len(docs)}")


class TracklistRAG:
    def __init__(self, index_dir: str):
        with open(os.path.join(index_dir, "docs.pkl"), "rb") as f:
            self.docs: List[Dict] = pickle.load(f)

        with open(os.path.join(index_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer: TfidfVectorizer = pickle.load(f)

        with open(os.path.join(index_dir, "matrix.pkl"), "rb") as f:
            self.matrix = pickle.load(f)

    def make_query(
        self,
        current_track: str,
        prev_track_1: Optional[str] = None,
        prev_track_2: Optional[str] = None,
        tracklist_hint: Optional[str] = None,
    ) -> str:
        return (
            f"tracklist {safe_str(tracklist_hint)} "
            f"prev2 {safe_str(prev_track_2)} "
            f"prev1 {safe_str(prev_track_1)} "
            f"current {safe_str(current_track)}"
        )

    def recommend_next(
        self,
        current_track: str,
        prev_track_1: Optional[str] = None,
        prev_track_2: Optional[str] = None,
        tracklist_hint: Optional[str] = None,
        retrieve_k: int = 100,
        recommend_k: int = 10,
        exact_current_boost: float = 1.8,
    ) -> List[Dict]:
        query = self.make_query(
            current_track=current_track,
            prev_track_1=prev_track_1,
            prev_track_2=prev_track_2,
            tracklist_hint=tracklist_hint,
        )

        q_vec = self.vectorizer.transform([query])

        # cosine similarity와 같은 역할 (TF-IDF L2 norm 기본 적용)
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        if len(scores) == 0:
            return []

        top_k = min(retrieve_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        current_norm = normalize_text(current_track)
        prev1_norm = normalize_text(prev_track_1)
        prev2_norm = normalize_text(prev_track_2)
        hint_norm = normalize_text(tracklist_hint)

        agg = defaultdict(lambda: {
            "score": 0.0,
            "count": 0,
            "examples": [],
        })

        for idx in top_idx:
            d = self.docs[int(idx)]
            sim = float(scores[int(idx)])
            weight = sim

            if normalize_text(d["current_track"]) == current_norm:
                weight *= exact_current_boost

            if prev1_norm and normalize_text(d.get("prev_track_1")) == prev1_norm:
                weight *= 1.20

            if prev2_norm and normalize_text(d.get("prev_track_2")) == prev2_norm:
                weight *= 1.10

            if hint_norm and hint_norm in normalize_text(d.get("tracklist_title")):
                weight *= 1.10

            nxt = d["next_track"]
            agg[nxt]["score"] += weight
            agg[nxt]["count"] += 1

            if len(agg[nxt]["examples"]) < 3:
                agg[nxt]["examples"].append(
                    {
                        "matched_current": d["current_track"],
                        "matched_prev1": d.get("prev_track_1"),
                        "matched_prev2": d.get("prev_track_2"),
                        "tracklist_title": d.get("tracklist_title"),
                        "similarity": round(sim, 4),
                    }
                )

        results = sorted(
            [
                {
                    "next_track": track,
                    "score": round(info["score"], 4),
                    "count": info["count"],
                    "examples": info["examples"],
                }
                for track, info in agg.items()
            ],
            key=lambda x: (-x["score"], -x["count"], x["next_track"].lower()),
        )

        return results[:recommend_k]


def cmd_build(args):
    build_index(args.tracks_csv, args.out_dir)


def cmd_recommend(args):
    rag = TracklistRAG(args.index_dir)
    recs = rag.recommend_next(
        current_track=args.current_track,
        prev_track_1=args.prev_track_1,
        prev_track_2=args.prev_track_2,
        tracklist_hint=args.tracklist_hint,
        retrieve_k=args.retrieve_k,
        recommend_k=args.recommend_k,
    )

    if not recs:
        print("추천 결과가 없습니다.")
        return

    print(f"\nCurrent track: {args.current_track}")
    if args.prev_track_1:
        print(f"Prev track 1: {args.prev_track_1}")
    if args.prev_track_2:
        print(f"Prev track 2: {args.prev_track_2}")
    if args.tracklist_hint:
        print(f"Tracklist hint: {args.tracklist_hint}")

    print("\nRecommended next tracks:")
    for i, r in enumerate(recs, start=1):
        print(f"{i:02d}. {r['next_track']} | score={r['score']} | count={r['count']}")
        for ex in r["examples"]:
            print(
                f"    - {ex['tracklist_title']} "
                f"(sim={ex['similarity']}, prev1={ex['matched_prev1']}, prev2={ex['matched_prev2']})"
            )


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build")
    p1.add_argument("--tracks-csv", required=True)
    p1.add_argument("--out-dir", default="tracklist_rag_index")
    p1.set_defaults(func=cmd_build)

    p2 = sub.add_parser("recommend")
    p2.add_argument("--index-dir", default="tracklist_rag_index")
    p2.add_argument("--current-track", required=True)
    p2.add_argument("--prev-track-1", default=None)
    p2.add_argument("--prev-track-2", default=None)
    p2.add_argument("--tracklist-hint", default=None)
    p2.add_argument("--retrieve-k", type=int, default=100)
    p2.add_argument("--recommend-k", type=int, default=10)
    p2.set_defaults(func=cmd_recommend)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)