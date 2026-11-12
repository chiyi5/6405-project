"""Evaluate course retrieval against complete, independently judged qrels.

--make-template writes blank labels for human annotation; it does NOT fabricate
gold relevance. --smoke uses author-written cases and is not a research benchmark.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.learning import LearningRecommender, parse_gaps

DEMO_CASES = [
    {"id": "demo-backend", "gaps": "SQL\nDocker\nFastAPI", "known": ["python"]},
    {"id": "demo-data", "gaps": "SQL joins\nSQL aggregation\nmodel evaluation", "known": ["python"]},
    {"id": "demo-newcomer", "gaps": "Python\nGit", "known": []},
    {"id": "demo-unsupported", "gaps": "Kubernetes\nRust", "known": []},
    {"id": "demo-python-overlap", "gaps": "Python", "known": []},
    {"id": "demo-api-test", "gaps": "FastAPI\nAPI development\nTesting", "known": ["python"]},
    {"id": "demo-priorities", "gaps": "Docker | MANDATORY\nPython | PREFERRED\nSQL | REQUIRED", "known": []},
    {"id": "demo-aliases", "gaps": "PostgreSQL\nSQL\nVersion control", "known": []},
    {"id": "demo-prerequisites", "gaps": "Python\nFastAPI\nTesting", "known": []},
    {"id": "demo-partial-catalog", "gaps": "Docker\nKubernetes", "known": []},
    {"id": "demo-already-known", "gaps": "Python\nGit", "known": ["python", "git"]},
    {"id": "demo-ml", "gaps": "Machine learning\nModel evaluation", "known": ["python"]},
]


def ndcg(retrieved, grades, k):
    def dcg(values):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(values))
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return dcg([grades[c] for c in retrieved[:k]]) / ideal if ideal else None


def metrics(result, grades, k):
    ids = [c["id"] for c in result["recommendations"]]
    positive = {c for c, grade in grades.items() if grade > 0}
    hits = [i + 1 for i, c in enumerate(ids[:k]) if c in positive]
    return {"ndcg_at_k": ndcg(ids, grades, k),
            "recall_at_k": len(set(ids[:k]) & positive) / len(positive) if positive else None,
            "mrr_at_k": 1 / hits[0] if hits else (0.0 if positive else None)}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No evaluation rows to write.")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--make-template", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--cases", type=Path, help="JSON list: id, gaps (one per line), known, split, group_id")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("results/learning"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-model")
    args = parser.parse_args()
    engine = LearningRecommender(dense_model=args.dense_model)
    cases = json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else DEMO_CASES
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("Case IDs must be unique.")
    if args.make_template:
        rows = [{"case_id": c["id"], "resource_id": r["id"], "relevance": "", "annotator": "", "notes": ""}
                for c in cases for r in engine.courses]
        write_csv(args.make_template, rows)
        print(f"Wrote {len(rows)} blank judgements. Blank != irrelevant; have humans label every row 0..3.")
        return
    if not args.smoke and (not args.qrels or not args.cases):
        parser.error("Research evaluation requires --cases and --qrels. Use --smoke only for author-written examples.")
    qrels = defaultdict(dict)
    if not args.smoke:
        groups = defaultdict(set)
        for c in cases:
            if c.get("split") not in {"dev", "test"} or not c.get("group_id"):
                parser.error("Each research case requires split (dev/test) and group_id (original JD/profile group).")
            groups[c["group_id"]].add(c["split"])
        if any(len(splits) > 1 for splits in groups.values()):
            parser.error("A source group appears in both dev and test.")
        with args.qrels.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key, cid = row["case_id"], row["resource_id"]
                if cid in qrels[key]:
                    parser.error("Duplicate judgements: adjudicate annotators into one consensus row first.")
                try:
                    grade = int(row["relevance"])
                except ValueError:
                    parser.error("Unlabelled qrels: blank relevance is not a zero.")
                if grade not in range(4) or not row.get("annotator", "").strip():
                    parser.error("Relevance must be 0..3 and annotator must be populated.")
                qrels[key][cid] = grade
        cases = [c for c in cases if c["split"] == args.split]
        expected = {r["id"] for r in engine.courses}
        if not cases:
            parser.error("No cases in selected split.")
        for c in cases:
            if set(qrels[c["id"]]) != expected:
                parser.error(f"Incomplete or foreign resource judgements for {c['id']}; judge the entire small catalog.")
    configurations = [("bm25", False), ("tfidf", False), ("hybrid", False), ("hybrid", True)]
    if args.dense_model:
        configurations += [("dense", False), ("hybrid_dense", False), ("hybrid_dense", True)]
    rows = []
    for method, optimize in configurations:
        for c in cases:
            result = engine.recommend(parse_gaps(c["gaps"]), method=method, optimize=optimize,
                                      top_k=args.top_k, known_skills=c.get("known", []))
            row = {"evaluation_kind": "synthetic_smoke_NOT_benchmark" if args.smoke else "human_judged_" + args.split,
                   "case_id": c["id"], "method": method, "backend": result["backend"], "optimized": optimize,
                   "weighted_topic_coverage": result["weighted_coverage"], "latency_ms": result["latency_ms"],
                   "redundancy_ratio": result["redundancy_ratio"],
                   "prerequisite_warnings": sum(bool(r["unmet_prerequisites"]) for r in result["recommendations"]),
                   "n_recommended": len(result["recommendations"])}
            # This checks provenance plumbing, NOT semantic faithfulness or human agreement.
            spans = [(q, r["source_excerpt"]) for r in result["recommendations"] for q in r["evidence"].values()]
            row["evidence_span_integrity"] = mean(q in source for q, source in spans) if spans else None
            if not args.smoke:
                row.update(metrics(result, qrels[c["id"]], args.top_k))
            rows.append(row)
    filename = "demo_smoke.csv" if args.smoke else f"human_{args.split}.csv"
    write_csv(args.output_dir / filename, rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["optimized"])].append(row)
    summary = []
    for (method, optimized), group in grouped.items():
        entry = {"evaluation_kind": group[0]["evaluation_kind"], "method": method,
                 "optimized": optimized, "cases": len(group)}
        for key in ("weighted_topic_coverage", "redundancy_ratio", "n_recommended", "prerequisite_warnings"):
            entry["mean_" + key] = round(mean(r[key] for r in group), 4)
        if not args.smoke:
            for key in ("ndcg_at_k", "recall_at_k", "mrr_at_k"):
                values = [r[key] for r in group if r[key] is not None]
                entry["mean_" + key] = round(mean(values), 4) if values else None
        summary.append(entry)
    write_csv(args.output_dir / filename.replace(".csv", "_summary.csv"), summary)
    print(f"Wrote {len(rows)} per-case rows to {args.output_dir / filename}")
    if args.smoke:
        print("No NDCG/Recall/MRR reported: these examples have no independent human relevance labels.")


if __name__ == "__main__":
    main()
