"""
Optional: evaluate the neural matchers on the human-annotated Vanetik & Kogan (2023) benchmark.
  git clone https://github.com/NataliaVanetik/vacancy-resume-matching-dataset.git
  python experiments/run_human_benchmark.py vacancy-resume-matching-dataset
Writes results/H_human_benchmark_neural.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import human_benchmark as hb, matching as mt  # noqa: E402
from src.pipeline import MatchingPipeline, PipelineConfig  # noqa: E402

V, C, R = hb.load_benchmark(sys.argv[1] if len(sys.argv) > 1 else "vacancy-resume-matching-dataset")
pipe = MatchingPipeline(PipelineConfig())
ms = [mt.TfidfMatcher(), mt.BM25Matcher(), mt.PipelineMatcher(pipe, "D6_our_pipeline (MiniLM)")]
for name in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]:
    ms += [mt.EmbeddingMatcher(name, "doc"), mt.EmbeddingMatcher(name, "maxsim")]
try:
    ms.append(mt.CrossEncoderMatcher())
except Exception as e:
    print("cross-encoder unavailable:", e)
ft = ROOT / "models" / "biencoder_fit"
if ft.exists():
    m = mt.EmbeddingMatcher(str(ft), "doc"); m.name = "D7_finetuned_biencoder"; ms.append(m)
ranks = hb.run_matchers(ms, V, C)
res = pd.DataFrame([hb.evaluate_rank_matrix(r, R, n) for n, r in ranks.items()])
(ROOT / "results").mkdir(exist_ok=True)
res.to_csv(ROOT / "results" / "H_human_benchmark_neural.csv", index=False)
print(pd.DataFrame([hb.human_agreement(R), hb.random_baseline()]).to_string())
print(res[["method", "spearman_consensus", "top1_consensus", "ndcg@5_consensus"]].to_string())
