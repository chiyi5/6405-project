# results — 最终结果索引

做 PPT 只用这里列出的文件。被取代的旧结果已删除（原始压缩包里仍有）。

## 汇总

| 模块 | 基准 | Part I baseline | 最好方法 | 来源文件 |
|---|---|---|---|---|
| A 数据 | 外部 fit 数据集（cnamuangtoun/resume-job-description-fit） | — | train 6241 对 / 280 个 JD，test 1759 对 / 71 个 JD，三类标签约 49% / 25% / 26%；实验用 3000 对训练、600 对评估（67 个 JD）；SkillSpan test 357 篇文档、2208 个金标准技能 | `A_hf_dataset_stats.csv`, `A_step1_info.json`, `A_eval_label_dist.csv`, `A_sample_pairs.csv`（每类 5 个示例） |
| B 技能抽取 | SkillSpan test，soft F1 | M4 词典+ESCO（Part I 默认）0.229；M1 ESCO 0.131 | M6 JobBERT 0.511 | `B_skill_extraction.csv`, `step8/B_skill_extraction_with_esco.csv` |
| C 需求分类 | 749 句（**LLM 标注**，单一标注者），按 JD 分组交叉验证，宏 F1 | C1 规则 0.474 | C3b TF-IDF+LR 0.651（C5 DistilBERT 0.634） | `C_requirement_clf.csv` |
| D 匹配 | 外部 fit 数据集 600 对，Spearman | D6 本系统 0.005（v3：0.014） | D7 微调 MiniLM 0.281；未训练方法中 BM25 0.143 [0.000, 0.286] | `D_matching.csv`, `D7_finetune.csv`, `S_fit_benchmark_spearman_ci.csv`, `step5/T3_fit_benchmark_v3.csv` |
| D 匹配 | TalentCLEF dev，10 个查询，MAP | D6 pipeline v3 0.795 | mpnet 整篇相似度 0.869；TF-IDF 0.857；本系统混合版 D8 0.837。mpnet、TF-IDF 与混合版的差距**不显著**（p = 0.075 / 0.13） | `step5/T1_talentclef_dev.csv`, `S_talentclef_paired_bootstrap_minilm.csv` |
| D 匹配 | 人工标注基准（Vanetik & Kogan 2023），共识标签 Spearman | D6 −0.268 | BM25 0.283；人与人一致性仅 0.171 | `H_human_benchmark_all.csv`, `H_human_agreement.csv`, `step5/T3_human_benchmark_v3.csv` |
| D 诊断 | 同上 | — | 完全不看简历的“职位偏好”预测器 0.573 → 标签主要反映职位，而非人岗匹配 | `H_vacancy_prior_diagnostic.csv` |
| E 打分 | fit 数据集，Spearman | E1 手工权重 −0.002 | E3a 逻辑回归 0.128 | `E_learned_scorer.csv`, `E_grid_search.csv` |
| E 消融 | TalentCLEF dev，MAP（MiniLM），配对重抽样检验 | 完整混合版 0.837 | 去掉 JD 技能列表解析 → 0.589（−0.248，95% CI [0.093, 0.436]，显著）；去掉整篇相似度 → 0.795（−0.042，CI [0.018, 0.073]，显著）；去掉精确匹配 / 语义匹配 → 差异不显著；整篇相似度权重 2.0 → 0.860（+0.023，显著，但是在同一批 10 个查询上选出来的） | `T_talentclef_ablation_minilm.csv`, `S_talentclef_paired_bootstrap_minilm.csv`, `T_talentclef_per_query_ap_minilm.csv` |
| F 公平性 | 姓名反事实，最大分数差 | 修复前 8.13（Jff10fb00） | 修复后 0（30 个 JD、80 个案例） | `F_name_fairness_multi_jd.csv`, `step7/step7_names_summary.csv` |
| F 公平性 | 性别措辞 | — | 0（80 个案例） | `step7/step7_gender_summary.csv` |
| F 鲁棒性 | 全文转大写（fit 基准 40 例），平均 / 最大分数差 | 修复前：D6 6.72 / 50.0，D8 4.85 / 33.5 | 修复后：D6 3.02 / 37.5，D8 2.24 / 25.3（减少约一半，但没有消除）；TalentCLEF 上 D8 分数有变化的简历 87.5% → 30% | `step7/step7_caps_per_case_summary.csv`, `step7/step7_caps_per_case_before_after.csv`, `step7/step7_caps_before_after.csv` |
| F 鲁棒性 | 其他扰动（缩写、展开、项目符号、段落顺序） | — | D6 全部不变；D8 打乱段落顺序最大 4.3、改项目符号最大 0.6 | `step7/step7_robustness_summary.csv` |

| ★ 学习推荐：课程检索 | 3,428 门课程，test 30 条查询（20 技能名 + 10 JD 句），银标签，nDCG@10 | 原 SkillBridge 词法混合（只有 12 个资源，无法评估） | LTR-LR 0.730 [0.630, 0.826]；融合+交叉编码器 0.719；BM25(dev 调参) 0.684；SBERT 0.62–0.63；微调双编码器 0.619；与 BM25 差异均不显著，GloVe 0.325、LDA 0.164 显著更差；Word2Vec 0.594（云端） | `learning/bench/L1_*.csv`, `fig_L1_retrieval.png`, `L4_finetune_grid.csv`, `learning/bench_cloud_cpu/` |
| ★ 学习推荐：学习路径 | 23 个真实 JD×简历差距画像，k=5，加权缺口覆盖率 | 逐缺口检索（原设计）BM25 0.755 | 改进版 xQuAD：TF-IDF 0.921、融合 0.916、BM25 0.894、Dense 0.880；6 种检索器中 5 种显著（Holm p ≤ 0.02）；消融（线性融合）：原版 xQuAD 0.817 → 改进版 0.916 | `learning/bench/L2_*.csv`, `fig_L2_*.png` |
| ★ 学习推荐：误差与标签审计 | 最佳系统 top-10 中被银标签判为不相关的命中 | — | 约 37% 在描述里原文提到该技能 → 银标签漏标，绝对分数偏低 | `learning/bench/L3_*.csv` |
| ★ 主体 step9：传统 vs 神经 | fit 600 对，Spearman [95% CI，按 JD 重抽样] | D1 TF-IDF 0.076、D2 BM25 0.143、D7 微调 MiniLM 0.281 | D12 Linear SVM 0.317 [0.188, 0.447]、随机森林 0.310；D13 微调交叉编码器 0.181（微调前 −0.075）；LSA/LDA/GloVe ≈ 0 | `step9/S9_matching_all.csv`, `S9_dev_selection.csv`, `S9_ce_finetune_grid.csv` |
| ★ 主体 step9：三分类 | 同上，准确率 / 宏 F1 | 多数类 0.333 / 0.167 | SVM 0.473 / 0.463 | `step9/S9_classifiers.csv`, `S9_confusion_*.csv` |
| ★ 主体 step9：效率与解释 | — | — | ms/对：BM25 1.3、LSA 1.6、SVM 2.7、CE(GPU) 7.3；LR 前 20 特征全来自 JD 一侧（职位偏好） | `step9/S9_efficiency.csv`, `S9_logreg_top_terms.csv`, `S9_lda_topics.csv`, `S9_occlusion_examples.csv` |

## 使用这些数字时的注意事项
- C 模块的金标准标签由 LLM（Claude）标注、未经人工复核（`requirements_labelled.csv` 的 `annotator` 列），报告时需说明。
- TalentCLEF dev 只有 10 个查询，“权重 2.0 最好”这类结论只能当作趋势，不能当作调好的参数。
- TalentCLEF 只有 10 个查询，显著性检验的功效很低：“mpnet / TF-IDF 与混合版差距不显著”的意思是**无法区分**，不是“一样好”。
- 之前在没装 sentence-transformers 的环境里得出的“TF-IDF 显著优于混合版”不成立，不要引用。
- 混合版 MAP：本地最终代码 0.837（`T_talentclef_ablation_minilm.csv`），`step5/T1_talentclef_dev.csv` 里是修复前代码的 0.839。只用 pipeline 的结果（0.795）和所有 baseline 两次完全一致，差别应来自语料中少数原本就是全大写的简历（大写修复只影响这类简历），结论不变。
- 名字偏差降为 0 的原因：名字所在的简历首行和联系方式被分到 `contact` 段，不参与打分（相当于盲筛），并不是模型本身学会了去偏。
- 原来的演示用差距报告（`F_example_*`）只抽出 1 条必需技能、得分 100 且证据不对应，已删除，演示例子待重新挑选。

## 大写问题修复（2026-11-07，`PipelineConfig.caps_fix`）
- 原因 1：fit 基准的简历没有换行，`normalize_layout` 靠“小写→大写”边界切分段落；全文转大写后这些边界消失 → 段落、技能、学历识别失败（D6 的变化全部来自 fit 基准）。
- 原因 2：姓名行过滤只认 `Tan Wei Ming`，不认 `TAN WEI MING` → 姓名留在整篇相似度的文本里（D8 在 TalentCLEF 上几乎每份都有小变化）。
- 修复：只对大部分字母为大写的简历生效——按已知段落标题和日期重新切分，并接受大写姓名行。正常大小写的简历结果与修复前逐分相同（本地 192 个组合 + step7 的 80 个真实案例验证）。
- 真实数据（step7，MiniLM）：fit 基准平均分数差 D6 6.72 → 3.02，D8 4.85 → 2.24；TalentCLEF 上 D8 改善 26 例、变差 5 例。D6 在 TalentCLEF 上有 2 例从不变变成有变化（JD 35129，最大 6.25 分）。
- 剩余差异：粘连 + 大写时信息确实丢失；另外修复后的大写版本有时比原来的版本解析得更准（demo 数据上看到：多识别出段落、不再把项目时间算成工作经验），反而与原分数差得更远。

- ★ 课程检索的 test 集只有 30 条查询，置信区间较宽；LTR 的提升不显著（Holm p = 0.35），不能写成 "LTR 优于 BM25"。
- ★ 最终神经运行在 Windows / Python 3.14 上，gensim 装不上，所以 Word2Vec 只有云端 CPU 运行的结果（`learning/bench_cloud_cpu/`）。
- ★ step9 的 D12 分类器同时看 JD 和简历的词；LR 权重显示它主要利用 JD 一侧信息，和 "职位偏好" 诊断一致，报告里不要写成 "学会了判断人岗匹配"。
- ★ `L4_finetune_grid.csv` / `S9_ce_finetune_grid.csv` 早期版本把 lr 四舍五入成 0.0，已修正（脚本也已修复）。
- ★ 学习路径的 λ 曲线和 xQuAD 变体是在同一批 23 个画像上比较的，属于敏感性分析；默认 λ=0.7 是事先定的。
- ★ 银标签来自 Coursera 的技能标签（含噪声，如剧本写作课被标了 "unix shells"），本组未做人工标注，没有 `L5_human_eval.csv`。

## 文件夹
- `learning/bench/`：★ 学习推荐模块的全部评测结果（L1–L5）
- `step5/`：TalentCLEF 2026 Task A dev 结果、fit 与人工基准上的 v3 系统（test 集的 TREC 提交文件已删除，需要时重跑 `experiments/step5_talentclef.py`）
- `step7/`：大规模公平性 / 鲁棒性测试（30 个 JD、80 个案例）
- `step8/`：ESCO baseline（M1/M2，Python 3.10 环境）
- `run_log_*.txt`：各轮运行日志
