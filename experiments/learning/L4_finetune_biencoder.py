"""L4 — Fine-tune a Sentence-BERT bi-encoder for skill -> course retrieval (GPU recommended).

    python experiments/learning/L4_finetune_biencoder.py
    python experiments/learning/L1_retrieval_benchmark.py --neural --finetuned models/course_biencoder

Training data: only the *train* split queries (skills disjoint from dev/test). Each
(query, relevant course text) is a positive; other courses in the batch are negatives
(MultipleNegativesRankingLoss, i.e. in-batch softmax / InfoNCE). A small grid over
learning rate x epochs is evaluated on the dev split (nDCG@10); the best model is saved.

Output: models/course_biencoder/, results/learning/bench/L4_finetune_grid.csv
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.learning import retrievers as R  # noqa: E402
from src.learning.benchmark import load_benchmark  # noqa: E402
from src.learning.ir_metrics import evaluate_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--lrs", default="1e-5,2e-5,5e-5")
    ap.add_argument("--epochs", default="1,3")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=40, help="positives sampled per query")
    ap.add_argument("--out", default=str(ROOT / "models" / "course_biencoder"))
    args = ap.parse_args()
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    corpus, queries, qrels, _ = load_benchmark()
    text = dict(zip(corpus.id, corpus.text))
    ids = list(corpus.id)
    train, dev = queries[queries.split == "train"], queries[queries.split == "dev"]
    rng = random.Random(0)
    examples = []
    for qid, q in zip(train.qid, train["query"]):
        pos = sorted(qrels[qid])
        for d in rng.sample(pos, min(args.max_pos, len(pos))):
            examples.append(InputExample(texts=[q, text[d][:1000]]))
    print(f"{len(examples)} training pairs from {len(train)} queries")

    rows, best = [], (-1, None)
    for lr in [float(x) for x in args.lrs.split(",")]:
        for ep in [int(x) for x in args.epochs.split(",")]:
            model = SentenceTransformer(args.base)
            model.max_seq_length = 256
            dl = DataLoader(examples, shuffle=True, batch_size=args.batch)
            model.fit(train_objectives=[(dl, losses.MultipleNegativesRankingLoss(model))], epochs=ep,
                      warmup_steps=int(0.1 * len(dl) * ep), optimizer_params={"lr": lr}, show_progress_bar=False)
            ret = R.DenseRetriever(model).fit(list(corpus.text))
            idx, _ = ret.rank(list(dev["query"]), 100)
            run = {qid: [ids[j] for j in row] for qid, row in zip(dev.qid, idx)}
            v = evaluate_run(run, qrels, dev.qid)["ndcg@10"].mean()
            rows.append({"lr": lr, "epochs": ep, "dev_ndcg@10": v})
            print(rows[-1])
            if v > best[0]:
                best = (v, model)
    out = ROOT / "results" / "learning" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / "L4_finetune_grid.csv", index=False)
    best[1].save(args.out)
    print("saved best model ->", args.out)


if __name__ == "__main__":
    main()
