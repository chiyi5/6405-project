# Group 32 — JD–Resume Matching（扩展版）

原来的 notebook = 一条流水线跑通 1 份 JD + 1 份简历。
扩展版 = **Part I 原系统作为 baseline** + **Part II 六个模块的对比实验**：数据集与标注、多方法对比、量化评估、消融、公平性测试和 Demo。
每个组员负责一个模块，各自有一张结果表，可以直接放进 PPT。

**最终结果汇总**：notebook 第 8 节；每个数字对应哪个文件见 `results/README.md`。被取代的旧结果已从 `results/` 删除。

## 目录结构

```
Group32_NLPproject_extended.ipynb   主 notebook（Part I 原代码 + Part II 实验）
app.py                              Gradio 演示界面（python app.py）
requirements.txt
src/
  pipeline.py          核心流水线（原 notebook 模块化 + bug 修复 + 批量排序 + gap report）
  nlp_backend.py       spaCy / sentence-transformers 可选，缺失时自动退回轻量方法
  data_utils.py        A：数据加载、标注表生成、κ 一致性、多数投票、匿名化
  skill_extraction.py  B：ESCO / 过滤后 ESCO / 词典 / LLM / JobBERT-SkillSpan + P/R/F1 评估
  requirement_clf.py   C：规则 / 规则+段落标题 / TF-IDF+LR / zero-shot NLI / DistilBERT
  matching.py          D：TF-IDF / BM25 / bi-encoder(doc, maxsim) / cross-encoder / 本系统 / 微调
  scoring.py           E：学习型打分器、阈值与权重网格搜索、消融实验
  talentclef.py        A：TalentCLEF 2026 Task A 数据加载与 MAP / NDCG 评估
  human_benchmark.py   A：人工标注基准（Vanetik & Kogan 2023）
  stats.py             A：置信区间、配对重抽样检验
  fairness.py          F：姓名反事实测试、性别措辞测试、鲁棒性扰动、错误分析、LLM 反馈
  llm_clients.py       可选 LLM 接口（OpenAI / Anthropic / 本地 HF 模型）
  plots.py             统一风格的报告图
  classic_nlp.py       ★ D9–D12：LSA / LDA / GloVe / Word2Vec 匹配器、4 种有监督 fit 分类器、句子遮挡解释
  learning/            第 9 节学习推荐模块
    benchmark.py       ★ 3,428 门课程语料、133 条查询、银标准标签、23 个真实差距画像
    retrievers.py      ★ 12 种检索器（BM25/TF-IDF/LSA/LDA/GloVe/Word2Vec/Dense/交叉编码器/RRF/LTR）
    planner.py         ★ 学习路径组单：拼接查询 / 逐缺口 / 改进版 xQuAD
    ir_metrics.py      ★ nDCG/MAP/MRR/Recall、bootstrap CI、配对随机化检验、Holm 校正
    course_search.py   ★ 大目录推荐的生产入口（showcase_app 的 /api/courses）
data/
  annotation_guidelines.md   标注规范（先读这个）
  demo/                      ⚠ 合成的演示数据，只用来测试代码能不能跑，结果不能写进报告
tests/
  smoke_test.py              一键测试所有模块（python tests/smoke_test.py）
  test_benchmark.py          ★ 指标、检索器、组单、分类器的单元测试（python -m unittest tests/test_benchmark.py）
  make_demo_data.py / build_notebook.py
experiments/                 实验脚本 step1–step9（见文末）
  learning/                  ★ 学习推荐模块实验 L1–L5 + run_all.py
scripts/windows/             Windows 启动脚本（原来在根目录；★ Run-New-Experiments.cmd 一键跑新实验）
docs/RUBRIC_MAPPING.md       ★ 课程评分标准 → 项目证据对照表
results/
  README.md                  最终结果索引：每个数字来自哪个文件
  step5/ step7/ step8/       各步骤的最终输出
```

## 2026-11 升级：新增内容一览

对照课程评分标准（`docs/RUBRIC_MAPPING.md`），这次升级做了两部分：

**第 9 节学习推荐：从 demo 变成完整子项目**（最终结果：本机 RTX 5060 运行，`results/learning/bench/`）
- 数据：3,428 门课程（Coursera 目录 + 12 个官方文档单元）；133 条查询（96 个技能名 + 37 句真实 JD 需求句）；23 个真实 JD × 简历差距画像；银标准标签来自 Coursera 技能标签，检索时不可见；按技能划分 train/dev/test。
- 检索（test 30 条查询，nDCG@10，16 个系统）：LTR 逻辑回归 0.730、融合 + 交叉编码器重排 0.719、BM25(dev 调参) 0.684、LSA 0.658、TF-IDF 0.649、Sentence-BERT（MiniLM / mpnet / bge）0.62–0.63、微调双编码器 0.619（Recall@100 0.549，最高之一）、GloVe 0.325、LDA 0.164；Word2Vec 0.594（云端运行，`bench_cloud_cpu/`）。只有 GloVe、LDA 与 BM25 的差异显著（Holm p = 0.0015）。结论：短技能名查询上词法方法很强，预训练稠密模型不占优；微调数据只有 75 条查询，dev 有提升（0.495 → 0.524）但没有迁移到未见过的 test 技能。
- 学习路径（k=5，加权缺口覆盖率）：BM25 下拼接查询 0.724 → 逐缺口 0.755 → 改进版 xQuAD 0.894；xQuAD 在 6 种检索器中 5 种显著更好（GloVe 不显著）。消融（线性融合检索器）显示关键改动是多样性项归一化：原版 xQuAD 0.817 → 改进版 0.916。
- 错误分析：37% 的 "不相关" 前 10 命中其实提到了该技能 → 银标签有漏标；双人盲标表已生成（`data/learning/human/`）。

**主体 A–F：补齐课程里的传统方法、效率和解释**（`results/step9/`，同样 600 对测试集）
- Spearman：有监督传统分类器最好——Linear SVM 0.317 [0.188, 0.447]、随机森林 0.310、NB 0.261、LR 0.199，超过此前最好的微调双编码器 D7（0.281）；新微调的交叉编码器 D13 从 −0.075 提到 0.181（lr 5e-5、2 epoch，dev 网格选出）。无监督的 LSA 0.066、LDA 0.080、GloVe 0.024 和零样本 transformer（−0.11 ~ −0.01）基本无效。
- 三分类：SVM 准确率 0.473、宏 F1 0.463（多数类基线 0.333 / 0.167）。
- 解释：LR 权重最大的前 20 个特征全部来自 JD 一侧（如 "hands-on"、"fintech"、"docker"），说明模型主要在学"哪类职位的样本更常被标 good fit"，与第 5 节的"职位偏好"诊断（不看简历 0.573）一致；LDA 中 web/java/sql 主题在 good-fit 简历中权重更高（0.125 vs 0.082）。
- 效率：BM25 1.3 ms/对、LSA 1.6、TF-IDF+SVM 2.7、交叉编码器（GPU）7.3；GloVe 拟合峰值内存 431 MB。

运行（项目根目录）：
```powershell
python experiments/learning/run_all.py            # 学习模块 CPU 部分，约 6 分钟，云端已跑过
python experiments/learning/run_all.py --neural   # + Sentence-BERT / 交叉编码器 / 微调（需要联网下载模型，建议 GPU）
python experiments/step9_classic_vs_neural.py     # 主体 step9（加 --finetune-ce 需 GPU）
python experiments/learning/L5_human_judgements.py score   # 两人标完 data/learning/human/ 之后
python -m unittest tests/test_benchmark.py tests/test_learning.py
```
或者双击 `scripts\windows\Run-New-Experiments.cmd`。

## 怎么运行

### 新增个人贡献：Notebook 第 9 节

学习资源推荐模块已经接入主 notebook 的 **第 9 节**。可在 VS Code / Jupyter 中直接运行该节，无需启动网页或桌面应用。优先读取 Part II 的匹配结果 `r`；独立运行时使用项目已有的合成 JD/简历和离线匹配后端。

也可以在项目根目录直接运行下面的命令；它只执行第 9 节，将真实运行输出保存回 notebook，并生成推荐表、来源证据和四组对比表：

```bash
python experiments/run_learning_section.py
```

产物位于 `results/learning/L_*`。这是推荐模块的实现与合成数据验证，不是人工标注测试集上的准确率。`learning_app.py`、`learning_desktop.py` 是额外演示入口，不是本项目主入口。

### 项目展示前端（SkillBridge）

- **实时分析**：双击 `scripts\windows\Start-Showcase.cmd`（或在根目录运行 `python showcase_app.py --open-browser`），打开 http://127.0.0.1:8877 。
- **离线预览**：双击根目录 `Showcase.html`，只能看界面和已有实验结果。修改 `frontend/` 后用 `python experiments/build_showcase.py` 重新生成。
- 本地网页打不开（`WinError 10013`）时：先运行 `scripts\windows\Diagnose-Web.cmd` 排查，或改用 `scripts\windows\Start-Desktop.cmd`（桌面版，不占端口）。
- 详细说明见 [docs/SHOWCASE.md](docs/SHOWCASE.md)。

### 测试

```bash
python tests/smoke_test.py                             # Part I/II 全部模块（合成数据）
python -m unittest discover -s tests -p "test_*.py"    # 学习模块、检索基准、展示前端
node tests/showcase_ui_smoke.cjs                       # 前端交互冒烟测试（需要 Node）
```

### 原项目 Part I / Part II

1. **Part I（原型）在本仓库中不运行，notebook 请从「Part II — 0. Setup」开始跑。** Part I 需要三样仓库里没有的东西：
   `Resume2.pdf`（真实个人简历，不提交）、`Courses.csv`（原型的学位/专业名称表）、`ojd-daps-skills`（只支持 Python 3.10/3.11）。
   它的效果在 Part II 中定量复现：原版 ESCO 抽取器由 `experiments/step8_esco.py` 在独立环境中运行（M1 / M4，第 3 节），
   原型流水线在 `src/pipeline.py` 中重写为 D6，作为第 5–8 节的 baseline。Part II 和第 9、10 节不依赖 Part I 的任何变量。
   （JD 示例在 `data/demo/JD_example1_ntu_AI.txt`。）
2. `pip install -r requirements.txt`（`gensim` 是可选的：Python 3.14 暂时装不上，没有它 Word2Vec 会自动跳过）
   ESCO baseline 还需要 `python -m spacy download en_core_web_sm`（`ojd-daps-skills` 建议用 Python 3.10，和你们原来的环境一致）。
3. 打开 `Group32_NLPproject_extended.ipynb`，默认 `USE_DEMO = True`，从上往下跑一遍，确认环境没问题。
4. 收集并标注好真实数据后改成 `USE_DEMO = False`；需要跑 transformer 模型时改 `RUN_HEAVY = True`（建议用 GPU）。

真实数据的文件放置：
```
data/jds/*.txt|pdf|docx
data/resumes/*.pdf|docx|txt
data/annotations/pairs_labelled.csv            (jd_id, resume_id, label, annotator, ...)
data/annotations/requirements_labelled.csv     (jd_id, sent_id, section_header, sentence, gold_label, annotator)
data/annotations/jd_skills_labelled.csv        (doc_id, skill, keep(1/0), annotator)
```
空白标注表由 notebook 第 2 节自动生成（`*_to_label.csv`），标完另存为上面的文件名即可。

## 分工与交付物（按实际完成的工作）

| 成员 | 负责环节 | 代码 | 主要结果（notebook 小节 / 结果文件） |
|---|---|---|---|
| **A** | 数据、基准与评测方法 | `src/data_utils.py`, `src/talentclef.py`, `src/human_benchmark.py`, `src/stats.py` | 4 个数据集与标注规范；MAP/NDCG 指标；按职位重抽样与配对重抽样检验；外部数据集标签分析（第 2 节；`H_human_agreement.csv`, `S_fit_benchmark_spearman_ci.csv`, `experiments/step5_talentclef.py`） |
| **B** | 技能抽取 | `src/skill_extraction.py`, `experiments/step8_esco.py` | JobBERT soft F1 0.511 vs Part I 原配置 0.229 vs ESCO 0.131；JobBERT 接入流程（第 3 节；`B_skill_extraction.csv`, `step8/`） |
| **C** | 职位描述理解 | `src/requirement_clf.py`, `src/pipeline.py`（版面修复、标题识别、技能列表解析） | 优先级分类宏 F1 0.63–0.65 vs 规则 0.474；技能列表解析使 TalentCLEF MAP +0.25（第 4 节；`C_requirement_clf.csv`, `T_talentclef_ablation_minilm.csv`） |
| **D** | 匹配模型 | `src/matching.py` | 8 种匹配器 + 微调双编码器在三个基准上的对比；反向任务的“职位偏好”诊断（第 5 节；`D_matching.csv`, `D7_finetune.csv`, `step5/T1_talentclef_dev.csv`, `H_vacancy_prior_diagnostic.csv`） |
| **E** | 打分与消融 | `src/scoring.py`, `src/pipeline.py`（混合打分）, `app.py`, `experiments/step6_ablation_minilm.py` | 学习型打分器；TalentCLEF 消融与权重敏感性；Gradio Demo（第 6 节；`E_learned_scorer.csv`, `T_talentclef_ablation_minilm.csv`, `app.py`） |
| **F** | 公平性、鲁棒性与错误分析 | `src/fairness.py`, `experiments/step7_fairness_v3.py` | 名字偏差 8.13 → 0（30 个职位复测）；鲁棒性与大写问题；错误分析（第 7 节；`F_name_fairness_multi_jd.csv`, `step7/`） |
| （待填） | 学习资源推荐 + SkillBridge 前端 | `src/learning/`, `experiments/learning/`, `showcase_app.py`, `frontend/` | 3,428 门课程检索基准、xQuAD 学习路径、错误分析（第 9 节；`results/learning/bench/`） |
| （待填） | 传统 NLP vs transformer | `src/classic_nlp.py`, `experiments/step9_classic_vs_neural.py` | Linear SVM Spearman 0.317，超过微调双编码器 0.281；效率与解释（第 10 节；`results/step9/`） |

## 展示结构建议（录屏 + PPT）
1. Motivation（Part I 原型暴露的问题：notebook 第 1 节）
2. Dataset & Annotation (A)
3. System Overview（流水线图）
4. Experiments: Skill Extraction (B) · Requirement Classification (C) · Matching (D) · Scoring & Ablation (E)
5. Fairness, Robustness & Error Analysis (F)
6. 传统 NLP vs transformer（第 10 节）· 学习资源推荐（第 9 节）
7. Live Demo（SkillBridge 前端）· Limitations · 分工说明（评分项对照见 `docs/RUBRIC_MAPPING.md`）

## 注意
- `data/demo/` 是编的合成数据，标签是人为判断的，**结果不能写进 PPT**。
- **需求优先级标签的来源**：`data/annotations/requirements_labelled.csv`（749 句）的 `gold_label` 由 LLM（Claude）按 `data/annotation_guidelines.md` 标注，`annotator` 列为 `claude`，单一标注者，**组内不做人工复核**。C 模块的分数衡量的是与这批 LLM 标签的一致程度，PPT 和分工说明里需要如实说明。
- 第 9 节课程检索用的是 Coursera 技能标签生成的银标准标签；`data/learning/human/` 的人工标注流程是可选的，本组没有进行。
- 简历包含个人信息：标注和提交前先用 `du.anonymise_resume` 去掉姓名和联系方式；不要把别人的简历原件交上去。
- API key 用环境变量传入，不要写进 notebook。

## 代码说明：Part I 与 src/
`Group32_NLPproject_extended.ipynb` 的 Part I 是最初的原型，原样保留，作为所有对比实验的 baseline；
当前系统以 `src/` 中的模块化代码为准（Part II 全部从 `src/` 导入）。两处有同名函数但行为不同，阅读时以 `src/pipeline.py` 为准。

## 实验脚本
- `experiments/step1_prepare.py`：下载公开数据集，生成标注表
- `experiments/step2_run_all.py`：模块 A–F 的全部主实验
- `experiments/step3_followup.py`：后续实验（JobBERT 接入流水线、修复在全量数据上的验证、扩大公平性测试、人工标注基准）
- `experiments/run_human_benchmark.py`：只跑人工标注基准（Vanetik & Kogan 2023）
- `experiments/step4_finish.py`：JobBERT 接入流水线（600 对）、大规模鲁棒性修复、JobBERT 版人工基准
- `experiments/step5_talentclef.py`：TalentCLEF 2026 Task A（专家标注的候选人排序基准）评估 + 生成测试集提交文件（TREC 格式，可提交到 Codabench）
- `experiments/step6_ablation_minilm.py`：用真实 MiniLM 重做 TalentCLEF 消融 + 配对显著性检验
- `experiments/step7_fairness_v3.py`：在最终系统上重做公平性和鲁棒性测试（30 个 JD、80 个案例）；同时输出大写问题修复前后对比 `step7_caps_before_after.csv`（逐例对比见 `step7_caps_per_case_before_after.csv`）
- `experiments/step8_esco.py`：在独立的 Python 3.10/3.11 环境中跑 Part I 原版 ESCO 抽取器，作为定量 baseline
- `experiments/step9_classic_vs_neural.py`：★ 传统 NLP（LSA/LDA/GloVe/Word2Vec/有监督分类器）vs transformer，同一 600 对；效率与解释；可选交叉编码器微调网格
- `experiments/learning/L1_retrieval_benchmark.py` … `L5_human_judgements.py`：★ 学习推荐模块的检索对比、组单评估、错误分析、双编码器微调、人工标注；`run_all.py` 按顺序全部运行
- `experiments/add_notebook_sections.py`：★ 把 9.3–9.7 和第 10 节写进 notebook（可重复运行）
- `experiments/make_pair_sheets.py`：生成组员人工标注用的 Excel 表（每对由 2 人标注）
- `step1_prepare.py --demo` 的输出写到 `results/demo_run/`，不会覆盖真实的 `results/A_*` 文件；但它会把 `data/hf/` 换成 demo 数据，跑真实实验前要重新运行一次不带 `--demo` 的 step1。
