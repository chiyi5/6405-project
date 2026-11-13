# 独立贡献：基于证据的学习资源推荐

## 2026-11 升级：大规模评测（先看这里）

旧版只有 12 个资源和自编案例。现在有完整的评测基准和实验，见 notebook 9.3–9.7、`results/README.md` 和 `docs/RUBRIC_MAPPING.md`：

| 部分 | 文件 | 要点 |
|---|---|---|
| 基准 | `src/learning/benchmark.py` | 3,428 门课程；96 个技能查询 + 37 个 JD 句查询；23 个真实差距画像；按技能划分 60/20/20 |
| 检索器 | `src/learning/retrievers.py` | BM25、TF-IDF（词/字符）、LSA、LDA、GloVe、Word2Vec、Dense、交叉编码器、RRF、线性融合、LTR |
| 组单 | `src/learning/planner.py` | 拼接查询 / 逐缺口 / 改进版 xQuAD（多样性项按未覆盖权重归一化 + 逐缺口候选池） |
| 实验 | `experiments/learning/L1`–`L5`, `run_all.py` | dev 调参 → test 报告；CI、配对检验、消融、误差分析、人工标注 |
| 生产入口 | `src/learning/course_search.py` | `CourseSearch().plan(gaps, k=5)`；每条推荐附课程描述原句作为证据；`POST /api/courses` |

视频演示建议增加：输入 SQL(REQUIRED) + Docker / Kubernetes(PREFERRED) + Tableau(REQUIRED)，逐缺口检索给出 4 门 SQL 课和 1 门 Tableau 课（Docker、Kubernetes 没覆盖，覆盖率 0.8）；xQuAD 给出 1 门 SQL、1 门 Tableau 和 3 门容器课程，覆盖率 1.0。

## 首选入口：原项目 Notebook 第 9 节

主入口为 `Group32_NLPproject_extended.ipynb` 的第 9 节。直接运行该节即可，不依赖 HTTP 服务。
它优先消费 Part II 的 `r["details"]`，独立运行时读取原项目已有的合成 JD/简历并调用 `src.pipeline.MatchingPipeline`。
`python experiments/run_learning_section.py` 可只执行这四个新增代码单元，将输出保存到主 notebook 及 `results/learning/L_*`。
下方网页、桌面与命令行入口是可选附加演示，不作为整合完成的依据。

## 运行

独立页面只需要 Python 3.10+，不需要安装项目原有的重型依赖：

```powershell
python learning_app.py
# 浏览器打开 http://127.0.0.1:8765
```

```powershell
python learning_cli.py --gaps "SQL,Docker,FastAPI" --known "Python"
python learning_cli.py --match-json match_result.json --output results/learning/my_plan.json
python -m unittest discover -s tests -p test_learning.py -v
python experiments/evaluate_learning.py --smoke
```

如果 python 命令不可用，请改用你自己虚拟环境里的 python（例如 `.venv\Scripts\python.exe`）。原项目 `python app.py` 的 Gradio 页面也增加了按钮；它需要原有全部依赖和模型。独立页面没有这些依赖。

## 与组员代码的边界

- 输入：`PIPE.match(jd, resume)` 的结果；`details` 可以是 DataFrame，也可以是 JSON 记录列表。
- 适配：仅转换分数低于 50 或未知的 Skill 行；不把学历、年限、资格等转为可通过课程解决的技能。
- 输出：资源列表、每个技能的原文证据、来源、未覆盖技能、覆盖率和规划顺序。
- 原有意图分类、抽取、匹配和打分不变。`app.py` 仅新增一个入口函数及 Gradio 按钮。
- `src/learning/integration.py:recommend_from_match` 可在 notebook 或现有服务直接使用。

```python
from src.learning.integration import recommend_from_match
match_result = PIPE.match(jd_text, resume_text)
plan = recommend_from_match(match_result, top_k=5)
```

## 数据与来源：不要把起步数据当作完整实验

`data/learning/catalog.json` 包含 12 个学习单元，来源为 Python、PostgreSQL、Docker、FastAPI、scikit-learn 和 Git 的官方文档/教程。一个教程可以分成多个单元，这不是 12 门独立认证课程，也不能当作两个独立评测数据集。

来源页面于 2026-11-12 核对。保留简短目录标题或原文摘录；description 是编者摘要。技能标签是编者标注，非自动验证的金标准。`evidence` 必须能逐字定位到 `source_excerpt`，但这种检查只能证明引用可追溯，不能证明教学深度、解释正确性或学完后的熟练程度。

所有课时均未知。设置时间预算后，未知课时资源会排除，不能用零小时或猜测课时填补。先修技能是编者的规划建议，而非官方硬性条件。不自动补入先修课程；找不到已安排的先修技能时展示提醒。

正式研究应扩展到约 100–200 个有充分描述的资源；记录版本、原文来源、课程/教程类型和人工标注来源。API 不会访问用户简历之外的第三方服务，独立页面不调用 LLM。

## 算法与解释

1. 合并技能别名和重复缺口，保留最高优先级。
2. 为每个缺口单独检索，按优先级聚合相关度，避免单个长查询掩盖小众技能。
3. 默认 hybrid = **BM25 + TF-IDF 的词法 RRF**，不是 Dense 检索。单个检索器的零匹配项不参加融合。
4. 只有具有对应人工技能证据标签的资源才能覆盖缺口；相似度本身不等于课程教会了该技能。
5. 贪心选择资源：

   `新增覆盖权重 + 0.15 × 归一化检索分数 − 0.25 × 重复覆盖权重 − 0.5 × 未满足先修技能数`

   优先级权重：MANDATORY=3，REQUIRED=2，UNSPECIFIED=1，PREFERRED=0.5。
   无新增覆盖时停止，不强行填满 Top-K。这是一个启发式，不保证全局最优；相似度和主题覆盖不是概率。
6. 在已选择集合内，尽量先排已满足先修条件的资源；输出缺失先修提醒，不谎称学习者掌握了先修技能。
7. 解释通过原文模板生成；没有使用 LLM，所以不会捏造课时、证书、价格或课程成效。

Dense 是独立可选后端：

```powershell
python learning_cli.py --gaps "SQL,Docker" --method hybrid_dense --dense-model sentence-transformers/all-MiniLM-L6-v2
python learning_app.py --dense-model sentence-transformers/all-MiniLM-L6-v2
```

需要安装 sentence-transformers 并可加载该模型；加载失败时明确报错，不降级后继续报告 Dense。当前验证环境没有这个依赖，因此 Dense 与原 Gradio 页面没有实跑验证。

## 实验设计

`experiments/evaluate_learning.py` 对比 BM25、TF-IDF、词法混合 RRF、混合 RRF + 覆盖去重。指定 Dense 模型后增加 Dense 和 Dense 混合实验。

`--smoke` 只有四个编者构造的演示案例，输出到 `results/learning/demo_smoke.csv`。不生成 NDCG/MRR/Recall，不能在报告中作为正式效果结论。

准备研究案例 JSON：

```json
[
  {"id":"case1", "group_id":"original_jd_1", "split":"dev", "gaps":"SQL\nDocker", "known":["python"]},
  {"id":"case2", "group_id":"original_jd_2", "split":"test", "gaps":"FastAPI\nTesting", "known":["python"]}
]
```

```powershell
python experiments/evaluate_learning.py --cases data/learning/cases.json --make-template data/learning/qrels_blank.csv
# 两人独立判断，再生成达成共识后的 qrels_consensus.csv
python experiments/evaluate_learning.py --cases data/learning/cases.json --qrels data/learning/qrels_consensus.csv --split test
```

标注规则：0=不相关，1=仅边缘相关，2=对部分缺口有用，3=直接对应主要缺口。必须填写 annotator。小语料中每个案例对全部资源做完整判断；空标签不能当成不相关。两个标注者的原始表分别保留，评测输入使用裁决后的唯一共识行。

同一个 group_id（原始 JD/简历画像）不能跨开发和测试。先在开发集决定权重、池大小与方法，再冻结并跑测试集。评测计算 NDCG@K、Recall@K、MRR@K；理想排序取全部已标注资源，不仅是召回的候选。全无相关资源的案例，相关性指标记为空并单独分析，不能当满分。

加权主题覆盖基于与算法共用的人工技能映射，只能作为集合性质，不能独立证明推荐质量。`evidence_span_integrity` 只是引用字符串检查，不等于人工评定的 faithfulness。耗时分为冷启动与热运行再比较；演示页的单次耗时不可作为性能结论。

## 个人贡献与录屏

建议 3 分钟演示：

1. 展示原 matcher 的 Docker/SQL/FastAPI 证据缺口。
2. 独立模块读取结果；展示推荐、来源原文与未覆盖技能。
3. 同一组输入开关覆盖去重，观察重复课程和覆盖变化。
4. 输入 Kubernetes 展示语料不支持时的明确未覆盖。
5. 展示 test 集上的检索 / 组单结果（银标准标签），解释一个成功和一个失败案例。

可以负责的贡献：任务定义、数据收集与复核、技能映射、检索融合、覆盖选择、实验、集成及解释。当前实现由 AI 辅助生成；不得将其描述为全部亲手独立编写。课程要求申报 AI 使用，并对纯 AI 生成代码设有扣分规则。请实际理解、修改和验证核心算法，记录自己的设计决定、人工标注和实验，不要仅改作者名。

## 文件位置

- 核心：`src/learning/engine.py`
- 原项目适配：`src/learning/integration.py`
- 独立页面：`learning_app.py` + `src/learning/index.html`
- 命令行：`learning_cli.py`
- 语料：`data/learning/catalog.json`
- 实验：`experiments/evaluate_learning.py`
- 正确性检查：`tests/test_learning.py`
