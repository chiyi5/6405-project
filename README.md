# EE6405 Group 32 — Explainable JD–Resume Matching & Learning-Resource Recommendation

可解释的人岗匹配与学习资源推荐系统。输入一份职位描述（JD）和一份或多份简历，系统会抽取技能与需求，给出逐条需求的匹配证据和总分，对候选人排序，并针对缺失的技能从 3,428 门课程中生成学习路径。

项目分两部分：**Part I** 是最初的原型，作为所有实验的 baseline；**Part II** 把原型重构为可配置的 `MatchingPipeline`，并在公开基准上对每个组件做量化对比、消融、公平性与鲁棒性测试。最后接入学习资源推荐和 SkillBridge 网页前端。

- 主 notebook：`Group32_NLPproject_extended.ipynb`
- 最终结果汇总：notebook 第 8 节；每个数字来自哪个文件见 [`results/README.md`](results/README.md)
- 评分标准对照：[`docs/RUBRIC_MAPPING.md`](docs/RUBRIC_MAPPING.md)

---

## 分工（最终版）

| 成员 | 负责环节 | Notebook 小节 | 主要代码 |
|---|---|---|---|
| **1号** | Part I 基线原型与流水线主干 | Part I；Part II 第 1 节 | Part I 原型；`src/pipeline.py`（`MatchingPipeline` 主干）、`src/nlp_backend.py` |
| **2号** | JD 理解与鲁棒性 | 第 3、4 节；第 7 节（鲁棒性部分） | `data/annotation_guidelines.md`、`src/skill_extraction.py`、`src/requirement_clf.py`、`experiments/step8_esco.py`；`src/pipeline.py`（版面解析、大写修复）、`src/fairness.py`（格式扰动） |
| **3号** | 匹配模型 | 第 5、10 节 | `src/matching.py`、`src/classic_nlp.py`、`src/talentclef.py`、`src/human_benchmark.py`、`experiments/step5_talentclef.py`、`experiments/step9_classic_vs_neural.py` |
| **4号** | 评测与打分 | 第 2、6 节；第 7 节（公平性与错误分析） | `src/data_utils.py`、`src/stats.py`、`src/scoring.py`、`src/fairness.py`（反事实测试）、`experiments/step6_ablation_minilm.py`、`experiments/step7_fairness_v3.py` |
| **5号** | 学习资源推荐 | 第 9 节 | `src/learning/`、`experiments/learning/`、`experiments/run_learning_section.py`、`learning_app.py` / `learning_cli.py` / `learning_desktop.py` |
| **6号** | 系统与展示 | 第 8 节；notebook 集成 | `showcase_app.py`、`frontend/`、`Showcase.html`、`app.py`（Gradio）、`tests/`、`experiments/build_showcase.py`、`experiments/add_notebook_sections.py`；视频演示 |

---

## 系统流程

```
JD / 简历（txt · pdf · docx）
   │  读取、清洗、版面解析、分段、匿名化
   ▼
技能抽取（词典 / ESCO / JobBERT）   需求句分类（必需 / 加分）   经验与学历抽取
   │
   ▼
逐条需求匹配（精确 + 语义）  +  整篇相似度  ──►  混合打分 + 证据句
   │
   ├──►  单份简历：分数、需求表、差距报告
   ├──►  多份简历：候选人排序
   └──►  缺失技能 ──► 课程检索 ──► xQuAD 组单 ──► 学习路径
```

## 主要结果

完整数字、置信区间和使用注意事项见 [`results/README.md`](results/README.md)。

| 环节 | 基准 / 指标 | Part I baseline | 最好结果 |
|---|---|---|---|
| 技能抽取 | SkillSpan test，soft F1 | 词典 + ESCO 0.229；ESCO 0.131 | JobBERT 0.511 |
| 需求分类 | 749 句（LLM 标注），宏 F1 | 规则 0.474 | TF-IDF + LR 0.651；DistilBERT 0.634 |
| 匹配 | fit 数据集 600 对，Spearman | 本系统 0.005 | Linear SVM 0.317 [0.188, 0.447]；微调 MiniLM 0.281 |
| 匹配 | TalentCLEF dev，MAP | pipeline 0.795 | mpnet 0.869、混合版 0.837（差距不显著） |
| 诊断 | 人工基准 | — | 不看简历的“职位偏好”预测器 0.573 → 标签主要反映职位 |
| 消融 | TalentCLEF，MAP | 完整混合版 0.837 | 去掉 JD 技能列表解析 → 0.589（显著） |
| 公平性 | 姓名反事实，最大分数差 | 8.13 | 0（30 个 JD、80 个案例） |
| 鲁棒性 | 全文大写，平均分数差 | 6.72 | 3.02（约减半） |
| 课程检索 | 3,428 门课程，test nDCG@10 | — | LTR 0.730；BM25 0.684；SBERT 0.62–0.63 |
| 学习路径 | 23 个差距画像，加权缺口覆盖率 | 逐缺口检索 0.755 | 改进版 xQuAD 0.894（BM25）/ 0.921（TF-IDF） |

---

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- `gensim` 可选：Python 3.14 暂时装不上，没有它 Word2Vec 会自动跳过。
- Part I 原版 ESCO 抽取器（`ojd-daps-skills`）只支持 Python 3.10 / 3.11，由 `experiments/step8_esco.py` 在独立环境中运行。
- transformer 实验建议用 GPU。

### 2. 运行 notebook

打开 `Group32_NLPproject_extended.ipynb`，**从「Part II — 0. Setup」开始运行**。

- Part I 需要仓库里没有的文件（`Resume2.pdf` 真实简历、`Courses.csv`、`ojd-daps-skills`），这里不运行；它的效果在 Part II 中定量复现（ESCO 抽取器见第 3 节，原型流水线作为 D6 baseline）。
- 默认 `USE_DEMO = True`，用合成数据检查环境；跑真实数据改为 `USE_DEMO = False`，跑 transformer 改 `RUN_HEAVY = True`。
- 只运行第 9 节并把输出写回 notebook：`python experiments/run_learning_section.py`

### 3. 演示界面

| 界面 | 启动方式 | 说明 |
|---|---|---|
| SkillBridge 前端 | `python showcase_app.py --open-browser`，或双击 `scripts\windows\Start-Showcase.cmd` | 打开 http://127.0.0.1:8877 |
| 离线预览 | 双击 `Showcase.html` | 只能看界面和已有结果；修改 `frontend/` 后用 `python experiments/build_showcase.py` 重新生成 |
| Gradio 演示 | `python app.py` | 粘贴 / 上传 JD 和简历 → 分数、需求表、差距报告、多简历排序 |
| 桌面版 | 双击 `scripts\windows\Start-Desktop.cmd` | 本地端口被拦（`WinError 10013`）时使用；排查用 `scripts\windows\Diagnose-Web.cmd` |

### 4. 测试

```bash
python tests/smoke_test.py                            # Part I/II 全部模块（合成数据）
python -m unittest discover -s tests -p "test_*.py"   # 指标、检索器、组单、前端路由
node tests/showcase_ui_smoke.cjs                      # 前端交互冒烟测试（需要 Node）
```

### 5. 复现实验

```bash
python experiments/step1_prepare.py                 # 下载公开数据集、生成标注表
python experiments/step2_run_all.py                 # 技能抽取、需求分类、匹配、打分、公平性主实验
python experiments/step5_talentclef.py              # TalentCLEF 2026 Task A
python experiments/step6_ablation_minilm.py         # 消融 + 配对显著性检验
python experiments/step7_fairness_v3.py             # 公平性与鲁棒性（30 个 JD、80 个案例）
python experiments/step9_classic_vs_neural.py       # 传统 NLP vs transformer（--finetune-ce 需 GPU）
python experiments/learning/run_all.py              # 学习推荐 L1–L4（CPU 部分）
python experiments/learning/run_all.py --neural     # + SBERT / 交叉编码器 / 微调（建议 GPU）
```

Windows 下也可双击 `scripts\windows\Run-New-Experiments.cmd`。

---

## 目录结构

```
Group32_NLPproject_extended.ipynb   主 notebook（Part I 原型 + Part II 实验）
showcase_app.py                     SkillBridge 前端服务
Showcase.html                       前端离线版
app.py                              Gradio 演示
learning_app.py / learning_cli.py / learning_desktop.py   学习推荐独立演示入口
diagnose_web.py                     本地网页连接诊断
requirements.txt
src/
  pipeline.py          MatchingPipeline 主干
  nlp_backend.py       spaCy / sentence-transformers，缺失时退回轻量方法
  skill_extraction.py  6 种技能抽取方法 + P/R/F1
  requirement_clf.py   5 种需求优先级分类方法
  matching.py          8 种匹配器 + 微调双编码器
  classic_nlp.py       LSA / LDA / GloVe / Word2Vec / 有监督分类器 / 遮挡解释
  talentclef.py        TalentCLEF 2026 数据与 MAP / NDCG
  human_benchmark.py   Vanetik & Kogan 人工标注基准
  data_utils.py        数据加载、标注表、一致性、匿名化
  stats.py             bootstrap 置信区间、配对检验
  scoring.py           学习型打分器、网格搜索、消融
  fairness.py          反事实公平性、格式扰动、错误分析
  llm_clients.py       可选 LLM 接口（API key 用环境变量）
  plots.py             报告图
  learning/            学习资源推荐（检索器、IR 指标、xQuAD 组单、接入匹配流程）
frontend/              SkillBridge 前端（中英双语）
tests/                 冒烟测试、单元测试、UI 测试、notebook 构建
experiments/           step1–step9 主实验；learning/ 下为 L1–L5
data/
  annotation_guidelines.md   需求句标注规范
  annotations/               749 句需求标注（LLM 标注）
  learning/                  课程语料、查询、银标签、差距画像
  demo/                      ⚠ 合成数据，只用于测试代码
docs/                  RUBRIC_MAPPING.md · LEARNING_MODULE.md · SHOWCASE.md
results/               最终结果（索引见 results/README.md）
scripts/windows/       Windows 一键启动 / 运行脚本
```

## 数据与模型

- **fit 数据集**：`cnamuangtoun/resume-job-description-fit`（Hugging Face），实验用 3,000 对训练、600 对评估（67 个 JD）
- **SkillSpan**：技能抽取评测（test 357 篇、2,208 个金标准技能）
- **TalentCLEF 2026 Task A**：专家标注的候选人排序基准
- **Vanetik & Kogan (2023)**：简历–职位人工标注基准
- **ESCO**：欧盟技能分类体系（Part I baseline）
- **Coursera 课程目录**：3,428 门课程 + 12 个官方文档单元
- 模型：spaCy `en_core_web_sm`、JobBERT、MiniLM / mpnet / bge 句向量、交叉编码器、DistilBERT、GloVe

下载的数据集和模型（`data/hf/`、`talentclef/`、`models/`）不提交，由实验脚本重新生成。

## 注意事项

- `data/demo/` 是合成数据，**结果不能写进报告或 PPT**。
- 749 句需求优先级标签由 LLM（Claude）按 `data/annotation_guidelines.md` 标注（`annotator = claude`），单一标注者，组内未做人工复核；需求分类的分数衡量的是与这批 LLM 标签的一致程度。
- 课程检索使用 Coursera 技能标签生成的银标准标签；`data/learning/human/` 的人工标注流程是可选的，本组没有进行。
- 名字偏差降为 0 是因为简历首行和联系方式被分到 `contact` 段不参与打分（相当于盲筛），不是模型学会了去偏。
- TalentCLEF dev 只有 10 个查询、课程检索 test 只有 30 条查询，“不显著”表示无法区分，不表示一样好。
- 简历含个人信息：提交前用 `data_utils.anonymise_resume` 去掉姓名和联系方式；不要提交别人的简历原件。
- 当前系统以 `src/` 为准；Part I 原型原样保留作 baseline，同名函数行为不同时以 `src/pipeline.py` 为准。
