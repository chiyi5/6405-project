# Cloud CPU run (Linux, Python 3.13, gensim 4.x) — 2026-11-13

Same scripts without `--neural`. Kept because it is the only run that includes the **Word2Vec**
retriever (gensim has no wheel for the Windows Python 3.14 environment used for the final neural run).
Deterministic systems (BM25, TF-IDF, LSA, GloVe, all L2 strategies on BM25/TF-IDF/LSA) match the final
run in `results/learning/bench/` exactly. Systems that include Word2Vec (LTR, linear fusion) differ, and
LDA's dev-selected k differed between platforms (k=100 here, k=200 on Windows: 0.176 vs 0.164 nDCG@10).
