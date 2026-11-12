"""Run an independent demo, or adapt a saved JD matcher result."""
import argparse
import json
from pathlib import Path

from src.learning import LearningRecommender, gaps_from_match, parse_gaps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--gaps", default=None, help="Comma-separated skills")
    source.add_argument("--match-json", type=Path, help="JSON object containing a details record list")
    p.add_argument("--method", choices=["bm25", "tfidf", "hybrid", "dense", "hybrid_dense"], default="hybrid")
    p.add_argument("--dense-model")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--known", default="")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--budget-hours", type=float)
    p.add_argument("--no-optimize", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    gaps = (gaps_from_match(json.loads(args.match_json.read_text(encoding="utf-8")))
            if args.match_json else parse_gaps((args.gaps or "SQL,Docker,FastAPI").replace(",", "\n")))
    engine = LearningRecommender(**({"catalog_path": args.catalog} if args.catalog else {}), dense_model=args.dense_model)
    result = engine.recommend(gaps, method=args.method, top_k=args.top_k, optimize=not args.no_optimize,
                              known_skills=[s for s in args.known.split(",") if s.strip()], budget_hours=args.budget_hours)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
