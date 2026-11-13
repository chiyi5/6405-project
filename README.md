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
data/
  annotation_guidelines.md   标注规范（先读这个）
  demo/                      ⚠ 合成的演示数据，只用来测试代码能不能跑，结果不能写进报告
tests/
  smoke_test.py              一键测试所有模块（python tests/smoke_test.py）
  make_demo_data.py / build_notebook.py
experiments/                 实验脚本 step1–step8（见文末）
results/
  README.md                  最终结果索引：每个数字来自哪个文件
  step5/ step7/ step8/       各步骤的最终输出
```

## 怎么运行

### 新增个人贡献：Notebook 第 9 节

学习资源推荐模块已经接入主 notebook 的 **第 9 节**。可在 VS Code / Jupyter 中直接运行该节，无需启动网页或桌面应用。优先读取 Part II 的匹配结果 `r`；独立运行时使用项目已有的合成 JD/简历和离线匹配后端。

也可以在项目根目录直接运行下面的命令；它只执行第 9 节，将真实运行输出保存回 notebook，并生成推荐表、来源证据和四组对比表：

```powershell
& D:\NLP\.venv\Scripts\python.exe experiments/run_learning_section.py
```

产物位于 `results/learning/L_*`。这是推荐模块的实现与合成数据验证，不是人工标注测试集上的准确率。`learning_app.py`、`learning_desktop.py` 是额外演示入口，不是本项目主入口。

### 原项目 Part I / Part II

1. Part I 需要原始的 `JD_example1_ntu_AI.txt`、`Resume2.pdf`、`Courses.csv`，**这三个文件不在本压缩包里**，运行前请放到项目根目录。
   （`Resume2.pdf` 含个人信息，提交代码时请勿附上原件。）
2. `pip install -r requirements.txt`
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

## 分工与交付物（2026-11-07 更新，按实际完成的工作）

| 成员 | 负责环节 | 代码 | 主要结果（notebook 小节 / 结果文件） |
|---|---|---|---|
| **A** | 数据、基准与评测方法 | `src/data_utils.py`, `src/talentclef.py`, `src/human_benchmark.py`, `src/stats.py` | 4 个数据集与标注规范；MAP/NDCG 指标；按职位重抽样与配对重抽样检验；外部数据集标签分析（第 2 节；`H_human_agreement.csv`, `S_fit_benchmark_spearman_ci.csv`, `experiments/step5_talentclef.py`） |
| **B** | 技能抽取 | `src/skill_extraction.py`, `experiments/step8_esco.py` | JobBERT soft F1 0.511 vs Part I 原配置 0.229 vs ESCO 0.131；JobBERT 接入流程（第 3 节；`B_skill_extraction.csv`, `step8/`） |
| **C** | 职位描述理解 | `src/requirement_clf.py`, `src/pipeline.py`（版面修复、标题识别、技能列表解析） | 优先级分类宏 F1 0.63–0.65 vs 规则 0.474；技能列表解析使 TalentCLEF MAP +0.25（第 4 节；`C_requirement_clf.csv`, `T_talentclef_ablation_minilm.csv`） |
| **D** | 匹配模型 | `src/matching.py` | 8 种匹配器 + 微调双编码器在三个基准上的对比；反向任务的“职位偏好”诊断（第 5 节；`D_matching.csv`, `D7_finetune.csv`, `step5/T1_talentclef_dev.csv`, `H_vacancy_prior_diagnostic.csv`） |
| **E** | 打分与消融 | `src/scoring.py`, `src/pipeline.py`（混合打分）, `app.py`, `experiments/step6_ablation_minilm.py` | 学习型打分器；TalentCLEF 消融与权重敏感性；Gradio Demo（第 6 节；`E_learned_scorer.csv`, `T_talentclef_ablation_minilm.csv`, `app.py`） |
| **F** | 公平性、鲁棒性与错误分析 | `src/fairness.py`, `experiments/step7_fairness_v3.py` | 名字偏差 8.13 → 0（30 个职位复测）；鲁棒性与大写问题；错误分析（第 7 节；`F_name_fairness_multi_jd.csv`, `step7/`） |

## 报告结构建议
1. Introduction & Motivation（Part I 原型暴露的问题：README 上方表格和 notebook 第 1 节）
2. Related Work（ESCO / SkillSpan / Sentence-BERT / cross-encoder / 招聘中的算法公平性）
3. Dataset & Annotation (A)
4. System Overview（流水线图）
5. Experiments: Skill Extraction (B) · Requirement Classification (C) · Matching (D) · Scoring & Ablation (E)
6. Fairness, Robustness & Error Analysis (F)
7. Demo · Limitations · Conclusion · 分工声明

## 注意
- `data/demo/` 是编的合成数据，标签是人为判断的，**结果不能写进报告或 PPT**。
- **需求优先级标签的来源**：`data/annotations/requirements_labelled.csv`（749 句）的 `gold_label` 由 LLM（Claude）按 `data/annotation_guidelines.md` 标注，`annotator` 列为 `claude`，单一标注者，未经组员人工复核。C 模块的分数衡量的是与这批 LLM 标签的一致程度，PPT 和分工声明里需要如实说明；如果时间允许，建议组员抽样复核一部分（例如 100 句）并报告一致率。
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
- `experiments/make_pair_sheets.py`：生成组员人工标注用的 Excel 表（每对由 2 人标注）
- `step1_prepare.py --demo` 的输出写到 `results/demo_run/`，不会覆盖真实的 `results/A_*` 文件；但它会把 `data/hf/` 换成 demo 数据，跑真实实验前要重新运行一次不带 `--demo` 的 step1。

## 项目展示前端（SkillBridge）

- **界面预览**：双击根目录 `Showcase.html`。文档默认留空，无内置候选人或固定匹配结果；分析自己的文档请使用下方实时入口。
- **实时分析**：双击 `Start-Showcase.cmd`，打开 http://127.0.0.1:8877 。支持自定义文档上传和推荐参数。
- 使用方法、演示顺序、模型配置与验证说明见 [docs/SHOWCASE.md](docs/SHOWCASE.md)。
- 离线展示页更新命令：`python experiments/build_showcase.py`。
