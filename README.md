# Group 32 — Explainable JD–Resume Matching（扩展版）

原来的 notebook = 一条流水线跑通 1 份 JD + 1 份简历。
扩展版 = **Part I 原系统作为 baseline** + **Part II 六个模块的对比实验**（数据集与标注、技能抽取、需求分类、匹配、打分与消融、公平性）+ **第 9 节学习资源推荐扩展** + **SkillBridge 展示前端**。
每个组员负责一个模块，各自有一张结果表，可以直接放进 PPT。

- **最终结果汇总**：notebook 第 8 节；每个数字对应哪个文件见 [`results/README.md`](results/README.md)。被取代的旧结果已从 `results/` 删除。
- **最终交付形式**：录屏视频 + PPT（不是书面报告），见文末「展示结构建议」。

## 快速开始

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm      # 可选：ESCO / spaCy 相关功能需要
python tests/smoke_test.py                   # 用合成数据检查所有模块能否跑通
```

然后打开 `Group32_NLPproject_extended.ipynb`，保持 `USE_DEMO = True` 从上往下跑一遍。

| 想做什么 | 命令（项目根目录） | Windows 双击 |
|---|---|---|
| 跑完整实验 notebook | 打开 `Group32_NLPproject_extended.ipynb` | — |
| 只跑第 9 节学习推荐 | `python experiments/run_learning_section.py` | — |
| Gradio 演示（成员 E） | `python app.py` | — |
| SkillBridge 展示前端（实时） | `python showcase_app.py --open-browser` → http://127.0.0.1:8877 | `Start-Showcase.cmd` |
| SkillBridge 展示前端（离线预览） | 直接打开 `Showcase.html` | `Showcase.html` |
| 学习推荐独立网页 | `python learning_app.py --open-browser` → http://127.0.0.1:8765 | `Start-Learning.cmd` / `Start-Learning.ps1` |
| 学习推荐桌面版（不用端口） | `python learning_desktop.py` | `Start-Desktop.cmd` |
| 学习推荐命令行 | `python learning_cli.py --gaps "SQL,Docker,FastAPI" --known "Python"` | — |

`.cmd` 启动器会依次查找项目内 `.venv`、上级目录 `.venv`，最后使用系统 `python`。

## 目录结构

```
Group32_NLPproject_extended.ipynb   主 notebook（Part I 原代码 + Part II 实验 + 第 9 节学习推荐）
app.py                              Gradio 演示界面（成员 E；含“生成学习路径”按钮）
showcase_app.py                     SkillBridge 展示前端的本地服务（端口 8877，读取 frontend/）
Showcase.html                       展示前端的离线预览（由 experiments/build_showcase.py 生成）
learning_app.py                     学习推荐独立网页（仅用标准库，端口 8765）
learning_desktop.py                 学习推荐桌面版（tkinter，无需浏览器和端口）
learning_cli.py                     学习推荐命令行入口
diagnose_web.py                     本地端口连不上时的诊断脚本（报告写到 results/learning/）
*.cmd / Start-Learning.ps1          Windows 一键启动 / 诊断脚本（见下文）
requirements.txt
src/
  pipeline.py          核心流水线（原 notebook 模块化 + bug 修复 + 批量排序 + gap report）
  nlp_backend.py       spaCy / sentence-transformers 可选，缺失时自动退回轻量方法
  data_utils.py        A：数据加载、标注表生成、κ 一致性、多数投票、匿名化
  talentclef.py        A：TalentCLEF 2026 Task A 数据加载与 MAP / NDCG 评估
  human_benchmark.py   A：人工标注基准（Vanetik & Kogan 2023）
  stats.py             A：置信区间、配对重抽样检验
  skill_extraction.py  B：ESCO / 过滤后 ESCO / 词典 / LLM / JobBERT-SkillSpan + P/R/F1 评估
  requirement_clf.py   C：规则 / 规则+段落标题 / TF-IDF+LR / zero-shot NLI / DistilBERT
  matching.py          D：TF-IDF / BM25 / bi-encoder(doc, maxsim) / cross-encoder / 本系统 / 微调
  scoring.py           E：学习型打分器、阈值与权重网格搜索、消融实验
  fairness.py          F：姓名反事实测试、性别措辞测试、鲁棒性扰动、错误分析、LLM 反馈
  llm_clients.py       可选 LLM 接口（OpenAI / Anthropic / 本地 HF 模型）
  plots.py             统一风格的报告图
  learning/            第 9 节：学习资源推荐
    engine.py            检索（BM25 / TF-IDF / RRF / 可选 Dense）+ 覆盖去重选择
    integration.py       从 PIPE.match() 结果生成学习计划（recommend_from_match）
    workflow.py          JD+简历 → 匹配 → 学习计划的完整流程，供网页/桌面/展示前端复用
    index.html           learning_app.py 的页面
frontend/              SkillBridge 展示前端（原生 HTML/CSS/JS，中英文切换：i18n.js / locales.js）
data/
  annotation_guidelines.md   标注规范（先读这个）
  annotations/               需求优先级标注（requirements_labelled.csv）、人工标注用的配对表
  learning/                  学习资源目录 catalog.json、案例与 qrels 模板
  demo/                      ⚠ 合成的演示数据，只用来测试代码能不能跑，结果不能写进 PPT
docs/
  LEARNING_MODULE.md         学习推荐模块：算法、数据来源、评测设计
  SHOWCASE.md                展示前端：使用方法、演示顺序、模型说明、验证
tests/
  smoke_test.py              一键测试所有模块（python tests/smoke_test.py）
  test_learning.py / test_showcase.py / showcase_ui_smoke.cjs   学习模块与展示前端测试
  make_demo_data.py / build_notebook.py
experiments/                 实验脚本 step1–step8 及第 9 节相关脚本（见下文）
results/
  README.md                  最终结果索引：每个数字来自哪个文件
  A_* … F_*, H_*, S_*, T_*   各模块最终结果
  step5/ step7/ step8/       各步骤的最终输出
  learning/                  第 9 节输出（L_*）与冒烟测试结果
```

## 怎么运行

### 原项目 Part I / Part II

1. Part I 需要原始的 `JD_example1_ntu_AI.txt`、`Resume2.pdf`、`Courses.csv`。`Resume2.pdf` 含个人信息，**不在仓库里**（`.gitignore` 已忽略所有 PDF），运行前请放到项目根目录。
2. `pip install -r requirements.txt`
   ESCO baseline（`ojd-daps-skills`）需要 Python 3.10/3.11，不装在主环境里，由 `experiments/step8_esco.py` 在独立环境中运行。
3. 打开 `Group32_NLPproject_extended.ipynb`，默认 `USE_DEMO = True`，从上往下跑一遍，确认环境没问题。
4. 换成真实数据后改成 `USE_DEMO = False`；需要跑 transformer 模型时改 `RUN_HEAVY = True`（建议用 GPU）。

真实数据的文件放置：
```
data/jds/*.txt|pdf|docx
data/resumes/*.pdf|docx|txt
data/annotations/pairs_labelled.csv            (jd_id, resume_id, label, annotator, ...)
data/annotations/requirements_labelled.csv     (jd_id, sent_id, section_header, sentence, gold_label, annotator)
data/annotations/jd_skills_labelled.csv        (doc_id, skill, keep(1/0), annotator)
```
空白标注表由 notebook 第 2 节自动生成（`*_to_label.csv`），标完另存为上面的文件名即可。

### 第 9 节：学习资源推荐（独立扩展）

学习资源推荐模块已经接入主 notebook 的 **第 9 节**，在 VS Code / Jupyter / PyCharm 中直接运行该节即可，不需要启动网页或桌面应用。它优先读取 Part II 的匹配结果 `r`；独立运行时使用项目已有的合成 JD/简历和离线匹配后端，不下载模型。

也可以只执行第 9 节，并把真实运行输出写回 notebook：

```bash
python experiments/run_learning_section.py
```

产物位于 `results/learning/L_*`（推荐表、来源证据、四组方法对比）。这是推荐模块的实现与合成数据验证，**不是人工标注测试集上的准确率**。算法、数据来源和评测设计详见 [docs/LEARNING_MODULE.md](docs/LEARNING_MODULE.md)。

在自己的代码里调用：

```python
from src.learning.integration import recommend_from_match
plan = recommend_from_match(PIPE.match(jd_text, resume_text), top_k=5)
```

`learning_app.py`、`learning_desktop.py`、`learning_cli.py` 是额外演示入口，不是本项目主入口。

### 项目展示前端（SkillBridge）

- **界面预览**：双击 `Showcase.html`。不内置候选人或固定匹配结果，可以切换中英文、查看实验结果；实时分析需要下面的服务。
- **实时分析**：双击 `Start-Showcase.cmd`（或 `python showcase_app.py --open-browser`），打开 http://127.0.0.1:8877 。支持粘贴/上传 TXT/PDF/DOCX、最多 12 份简历排序、学习路径生成与导出。
- 实时版匹配使用词典技能 + 离线字符 n-gram，**不是** MiniLM / JobBERT 实验配置，页面上有说明。
- 修改 `frontend/` 后重新生成离线预览：`python experiments/build_showcase.py`。
- 使用方法、演示顺序、模型配置与验证说明见 [docs/SHOWCASE.md](docs/SHOWCASE.md)。

### 本地网页打不开时

部分 Windows 环境会拦截本地端口（报错 `WinError 10013`）。

1. 双击 `Diagnose-Web.cmd`：分别测试标准库 HTTP 服务和本项目页面，报告写到 `results/learning/connection_diagnosis.txt`。
2. 服务窗口开着时双击 `Check-Connection.cmd`：检查 8765 端口监听并绕过代理直连。
3. 仍然不行就用 `Start-Desktop.cmd`（桌面版，不占端口），或用 `Showcase.html` 只做界面预览。

## 测试

```bash
python tests/smoke_test.py                             # Part I/II 全部模块（合成数据）
python -m unittest discover -s tests -p "test_*.py"    # 学习模块 + 展示前端 HTTP 测试
node tests/showcase_ui_smoke.cjs                       # 前端交互冒烟测试（需要 Node）
python experiments/evaluate_learning.py --smoke        # 学习推荐冒烟测试（不是正式评测）
```

## 分工与交付物（按实际完成的工作）

| 成员 | 负责环节 | 代码 | 主要结果（notebook 小节 / 结果文件） |
|---|---|---|---|
| **A** | 数据、基准与评测方法 | `src/data_utils.py`, `src/talentclef.py`, `src/human_benchmark.py`, `src/stats.py` | 4 个数据集与标注规范；MAP/NDCG 指标；按职位重抽样与配对重抽样检验；外部数据集标签分析（第 2 节；`H_human_agreement.csv`, `S_fit_benchmark_spearman_ci.csv`, `experiments/step5_talentclef.py`） |
| **B** | 技能抽取 | `src/skill_extraction.py`, `experiments/step8_esco.py` | JobBERT soft F1 0.511 vs Part I 原配置 0.229 vs ESCO 0.131；JobBERT 接入流程（第 3 节；`B_skill_extraction.csv`, `step8/`） |
| **C** | 职位描述理解 | `src/requirement_clf.py`, `src/pipeline.py`（版面修复、标题识别、技能列表解析） | 优先级分类宏 F1 0.63–0.65 vs 规则 0.474；技能列表解析使 TalentCLEF MAP +0.25（第 4 节；`C_requirement_clf.csv`, `T_talentclef_ablation_minilm.csv`） |
| **D** | 匹配模型 | `src/matching.py` | 8 种匹配器 + 微调双编码器在三个基准上的对比；反向任务的“职位偏好”诊断（第 5 节；`D_matching.csv`, `D7_finetune.csv`, `step5/T1_talentclef_dev.csv`, `H_vacancy_prior_diagnostic.csv`） |
| **E** | 打分与消融 | `src/scoring.py`, `src/pipeline.py`（混合打分）, `app.py`, `experiments/step6_ablation_minilm.py` | 学习型打分器；TalentCLEF 消融与权重敏感性；Gradio Demo（第 6 节；`E_learned_scorer.csv`, `T_talentclef_ablation_minilm.csv`, `app.py`） |
| **F** | 公平性、鲁棒性与错误分析 | `src/fairness.py`, `experiments/step7_fairness_v3.py` | 名字偏差 8.13 → 0（30 个职位复测）；鲁棒性与大写问题；错误分析（第 7 节；`F_name_fairness_multi_jd.csv`, `step7/`） |
| **扩展** | 学习资源推荐 + SkillBridge 前端 | `src/learning/`, `learning_*.py`, `showcase_app.py`, `frontend/`, `experiments/evaluate_learning.py` | 覆盖去重使冗余比 0.40 → 0.00、推荐单元 5 → 3（合成案例，非正式评测）（第 9 节；`results/learning/L_*`） |

## 展示结构建议（录屏 + PPT）

1. Motivation：Part I 原型暴露的问题（notebook 第 1 节）
2. Dataset & Annotation (A)
3. System Overview（流水线图）
4. Experiments：Skill Extraction (B) · Requirement Classification (C) · Matching (D) · Scoring & Ablation (E)
5. Fairness, Robustness & Error Analysis (F)
6. Extension：学习资源推荐（第 9 节）
7. Live Demo（SkillBridge 前端）· Limitations · 分工说明

## 注意

- `data/demo/` 是编的合成数据，标签是人为判断的，**结果不能写进 PPT**。
- **需求优先级标签的来源**：`data/annotations/requirements_labelled.csv`（749 句）的 `gold_label` 由 LLM（Claude）按 `data/annotation_guidelines.md` 标注，`annotator` 列为 `claude`，单一标注者，**组内不做人工复核**。C 模块的分数衡量的是与这批 LLM 标签的一致程度，PPT 和分工说明里需要如实说明。
- 第 9 节学习推荐目前只有合成冒烟案例的结果（`demo_smoke*.csv` 中标记为 `synthetic_smoke_NOT_benchmark`），不能当作正式效果结论。
- 简历包含个人信息：标注和提交前先用 `du.anonymise_resume` 去掉姓名和联系方式；不要把别人的简历原件交上去。
- API key 用环境变量传入，不要写进 notebook。
- 课程要求申报 AI 使用：PPT / 分工说明中请如实说明哪些部分由 AI 辅助完成。

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
- `experiments/make_pair_sheets.py`：生成人工标注用的 Excel 表（每对由 2 人标注）
- `step1_prepare.py --demo` 的输出写到 `results/demo_run/`，不会覆盖真实的 `results/A_*` 文件；但它会把 `data/hf/` 换成 demo 数据，跑真实实验前要重新运行一次不带 `--demo` 的 step1。

第 9 节与展示前端：

- `experiments/run_learning_section.py`：只执行 notebook 第 9 节并把输出写回 notebook，生成 `results/learning/L_*`
- `experiments/integrate_learning_notebook.py`：把第 9 节的单元格追加到主 notebook（不改动已有单元格）
- `experiments/evaluate_learning.py`：BM25 / TF-IDF / RRF / RRF+覆盖去重 的对比评测；`--smoke` 为合成冒烟测试，`--make-template` 生成空白 qrels 供人工判断
- `experiments/build_showcase.py`：根据 `frontend/` 重新生成离线 `Showcase.html`
