# SkillBridge 项目展示前端

## 界面语言

首次打开默认使用英文。右上角 **Language** 下拉框可切换 **English / 中文**，选择会保存在当前浏览器，下次打开自动恢复；浏览器禁用本地存储时，本次切换仍然有效。

切换语言保留当前文档、候选人选择、匹配结果、推荐方案和检索参数。界面标签、提示、结果说明和输入占位文字随语言切换；原始岗位/简历内容、课程标题与来源证据保持原文。实时版和离线 `Showcase.html` 都支持切换，无需联网获取翻译。

## 文档输入与界面预览

页面默认不填充岗位、简历或学习目标，没有示例下拉框或自动加载按钮。用户在匹配页粘贴或上传 JD 和简历；候选人排序页仅包含自己上传的文件。

双击 `Showcase.html` 可以预览界面、切换语言和查看已有实验结果与资源目录。该文件不再内嵌示例文档或预计算的匹配结果；可以填写文本并读取 TXT，但实时分析及 PDF/DOCX 解析需要下方的 Python 服务。页面会明确提示启动方式，不会用固定结果替代用户输入。

更新界面后可重新生成预览：

```powershell
python experiments/build_showcase.py
```

## 实时分析自定义文档

双击 **scripts\windows\Start-Showcase.cmd**，保持启动窗口打开，访问 http://127.0.0.1:8877 。

也可自行指定 Python 与端口：

```powershell
python showcase_app.py --port 8877 --open-browser
```

启动器优先使用项目 `.venv`，其次使用父目录 `.venv`，最后使用系统 Python。
基础依赖：`numpy pandas scikit-learn`；PDF / DOCX 上传还需要 `pypdf python-docx`。完整依赖见 `requirements.txt`。界面使用原生 HTML/CSS/JS，不需要 Node、前端打包或外网字体。

实时版支持：粘贴英文岗位与简历、上传 TXT/PDF/DOCX、单次最多 12 份简历排序、从差距生成学习路径、手动目标与预算、四种策略比较、导出 Markdown 学习报告。单个文档最多 5 MB、提取文本最多 40000 字符；扫描 PDF 需先 OCR。

## 模型与结果说明

- 实时匹配调用 `src.learning.workflow.match_documents` → 原 `MatchingPipeline`，使用词典技能和离线 hashed character n-grams，不下载模型。页面明确说明它不是 MiniLM / JobBERT 实验配置。
- 学习推荐调用 `LearningRecommender`，Hybrid 是 BM25 + TF-IDF 的 RRF 融合；主题覆盖不等于能力掌握。
- 实验页直接读取 `results/B_skill_extraction.csv`、`C_requirement_clf.csv`、`D_matching.csv`、`T_talentclef_ablation_minilm.csv`，以及 2026-11 升级新增的三个面板：传统 vs 神经（`step9/S9_matching_all.csv`）、课程检索（`learning/bench/L1_leaderboard.csv`）、学习路径（`learning/bench/L2_leaderboard_k5.csv`）；不同面板的基准和指标不同。
- `POST /api/courses` 提供 3,428 门课程目录上的学习路径推荐（BM25 + 归一化 xQuAD，见 notebook 9.7）；前端页面目前不调用它。
- 分类标签来自单一 LLM 标注者，尚未人工复核；TalentCLEF dev 只有 10 个查询。
- 输入文档不写入项目目录；PDF/DOCX 解析临时文件随请求完成清理。仅监听 127.0.0.1。

## 验证

```powershell
python -m unittest discover -s tests -p "test_*.py"
node tests/showcase_ui_smoke.cjs
```

HTTP 测试使用项目已有的内存 socket 传输方式，验证实际请求解析、路由、上传、排序与推荐；不证明当前机器的端口可连接。JS 测试使用轻量 DOM 模型，验证空白初始状态、自定义文档提交与界面流程，不代替浏览器视觉检查。

如当前环境无法连接本地端口，`Showcase.html` 仅可用于界面预览，不能分析自定义文档。
