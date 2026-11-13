# EE6405 评分标准对照表（Appendix 3）

每一项评分标准对应到项目里的具体证据（文件 / notebook 小节），可以直接用于报告、PPT 和视频脚本。
标 ★ 的是 2026-11 这次升级新增的内容。

| 评分项 | 4 分（Exemplary）要求 | 本项目的证据 |
|---|---|---|
| **Requirements**（≥2 种算法等） | 满足全部要求 | 匹配：8 种匹配器 + 微调（D1–D7），★ 新增 LSA / LDA / GloVe / Word2Vec / 4 种有监督分类器（D9–D12）；技能抽取 6 种（M1–M6）；需求分类 5 种（C1–C5）；★ 课程推荐 12 个检索系统 + 3 种组单策略 |
| **Choice of application & datasets** | 不同的应用 + 多个数据集 | 人岗匹配 + 可解释差距报告 + ★ 学习路径推荐。数据集：fit 数据集（8,000 对）、SkillSpan、TalentCLEF 2026、Vanetik & Kogan 人工基准、749 句需求标注、★ Coursera 课程目录（3,416 门）+ 官方文档 12 单元 |
| **Framework** | 好的模型选择 + 提升性能的额外努力 | JobBERT 技能抽取（F1 0.229 → 0.511）；混合打分；微调 MiniLM（Spearman −0.01 → 0.281）；★ 学习排序 LTR；★ 改进版 xQuAD（线性融合检索器上覆盖率 0.817 → 0.916，归一化是关键，见 `L2_ablation.csv`）；★ 微调课程双编码器（L4，lr×epoch 网格，dev 0.495 → 0.524）；★ 交叉编码器微调（step9，Spearman −0.075 → 0.181）；★ TF-IDF 对特征 + Linear SVM 达到 0.317，为主体匹配任务的新最好成绩 |
| **Experiment design** | 多个相关指标 + 与强 baseline 对比 | Spearman / AUC / macro-F1 / NDCG@5 / MAP；按 JD 聚类的 bootstrap 置信区间；配对重抽样检验。★ 课程检索：nDCG@10、MAP@100、MRR@10、P@5、Recall@10/100 + 95% CI + Holm 校正的配对随机化检验；dev 上调参、test 上报告，按技能划分防泄漏。★ 学习路径：加权覆盖率、必需技能覆盖、precision、冗余度 |
| **Interpretation** | 深入解释，并能推广到其他场景、说明物理意义 | 错误分析（B/C/F）、公平性反事实、"职位偏好" 诊断。★ LDA 主题（web/java/sql 主题在 good-fit 简历中更高）、LR 词权重（前 20 全是 JD 侧特征 → 模型学到的是职位偏好，可推广为：此类公开 fit 数据集的标签主要反映职位而非人岗匹配）、句子遮挡解释（哪些简历句子决定分数）；★ 银标签审计（37% 的 "错误" 命中其实提到了该技能 → 标签漏标）；★ 按查询长度 / 相关文档数分桶；★ xQuAD λ 的相关性-多样性权衡曲线 |
| **Readability** | 易读、组织好、有文档 | `src/` 模块化，每个文件有 docstring 说明对应课程知识点；README、results/README（每个数字的来源）、本对照表；★ 根目录启动脚本移到 `scripts/windows/` |
| **Efficiency** | 高效且不损失可读性 | ★ `L1_efficiency.csv`：每查询毫秒数、索引大小、建索引时间；★ `S9_efficiency.csv`：BM25 1.3 ms/对、LSA 1.6、SVM 2.7、交叉编码器（GPU）7.3，GloVe 峰值内存 431 MB；BM25 用稀疏矩阵一次算完所有查询（3,428 门课 < 0.1 ms/查询） |
| **Reusability** | 多个部分可复用 | 统一接口：`Retriever.fit/score/rank`、matcher 的 `fit/score_pairs`；`CourseSearch.plan()` 一行调用；`showcase_app.py` 提供 `POST /api/courses`；★ 单元测试 `tests/test_benchmark.py`（14 个）+ `tests/test_learning.py`（20 个） |
| **Presentation** | 出色的演示 + 直观的前端 | `showcase_app.py` 本地网页：匹配、候选人排序、学习路径、实验页；★ 实验页新增 "传统 vs 神经 / 课程检索 / 学习路径" 三个标签 |
| **Video skills** | 全员参与、流程顺畅 | 见 README "分工与交付物"，每个模块一人讲；建议的学习路径演示脚本见 `docs/LEARNING_MODULE.md` |

## 和课程内容的对应

| 课程知识点 | 在项目里的位置 |
|---|---|
| Preprocessing / BoW | `src/pipeline.py` 版面修复、`retrievers.tokenize`（保留 C++、C#、Node.js） |
| Term weighting | TF-IDF（D1）、BM25（D2，★ k1/b 在 dev 上调参） |
| Feature extraction | ★ LSA（D9、课程检索） |
| Topic modelling | ★ LDA（D10、课程检索；主题可视化） |
| ML classifiers | C3 TF-IDF+LR、E3 学习型打分器、★ D12 LR / SVM / NB / RF、★ LTR |
| Evaluation metrics | 见上 Experiment design |
| Word embeddings | ★ GloVe-100、★ 在语料上训练的 Word2Vec skip-gram |
| Attention / Transformers / PLMs | MiniLM、mpnet、bge、cross-encoder、JobBERT |
| Fine-tuning / hyper-parameter tuning | D7 微调双编码器、★ L4 课程双编码器（lr × epochs 网格）、★ step9 交叉编码器网格、M6 JobBERT 微调 |
| Applications | 匹配、需求分类、技能抽取（NER）、推荐 |

## 诚实说明（报告里要写）

- 课程检索的相关性标签来自 Coursera 自带的技能标签（银标准），L3 审计显示有漏标，所以绝对分数偏低；系统之间的比较仍然有效。本组未做人工相关性标注；`data/learning/human/` 的双盲标注表和 `L5_human_judgements.py` 只是预留的可选流程，没有对应结果。
- 课程检索 test 集只有 30 个查询：LTR（0.730）比 BM25（0.684）高，但 Holm 校正后不显著；只有 GloVe 和 LDA 显著更差。微调双编码器 dev 上有提升，test 上（0.619）没有超过原版 MiniLM（0.634），原因是训练只有 75 条查询、test 技能都没见过。
- Word2Vec 结果来自云端 CPU 运行（`results/learning/bench_cloud_cpu/`），其余为本机 GPU 运行。
- 学习路径评估有 23 个画像，xQuAD 在 6 种检索器中 5 种显著优于另外两种策略（Holm 校正后 p ≤ 0.02），GloVe 不显著。λ 曲线是在同一批画像上画的，是敏感性分析，不是调参结果。
- 课程要求申报 AI 使用：本次升级的代码由 AI 辅助编写，组员需要读懂并能讲解核心算法，按实际情况填写声明。
