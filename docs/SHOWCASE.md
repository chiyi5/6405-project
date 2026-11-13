# SkillBridge 项目展示前端

## 直接演示（无需安装或启动服务）

双击项目根目录的 **Showcase.html**，用 Chrome / Edge 打开。

包含 3 个合成岗位、8 份合成简历的 24 组匹配结果，由项目 Python 流水线实际计算并内嵌。学习推荐同样由原推荐器预先生成。页面明确标识离线案例演示，输入框与自定义参数锁定；结果不是实时计算，也不是研究评测结果。

推荐演示顺序：

1. 匹配分析：点击「开始匹配分析」，展示分数与逐条简历证据。
2. 点击「生成我的学习路径」，展示技能差距对应的官方资源及来源片段。
3. 候选人排序：选择内置简历，排序后点击候选人回看证据。
4. 实验对比：切换技能抽取、需求分类、匹配与消融面板，展开原始 CSV 数据。
5. 资源目录：搜索 SQL / Docker 等技能，打开官方教程。

更新原算法、数据或界面后，重新构建离线文件：

```powershell
python experiments/build_showcase.py
```

只收录 `data/demo/` 的合成案例，不读取私人简历。

## 实时分析自定义文档

双击 **Start-Showcase.cmd**，保持启动窗口打开，访问 http://127.0.0.1:8877 。

也可自行指定 Python 与端口：

```powershell
& D:\NLP\.venv\Scripts\python.exe showcase_app.py --port 8877 --open-browser
```

启动器优先使用项目 `.venv`，其次使用父目录 `.venv`，最后使用系统 Python。
基础依赖：`numpy pandas scikit-learn`；PDF / DOCX 上传还需要 `pypdf python-docx`。完整依赖见 `requirements.txt`。界面使用原生 HTML/CSS/JS，不需要 Node、前端打包或外网字体。

实时版支持：粘贴英文岗位与简历、上传 TXT/PDF/DOCX、单次最多 12 份简历排序、从差距生成学习路径、手动目标与预算、四种策略比较、导出 Markdown 学习报告。单个文档最多 5 MB、提取文本最多 40000 字符；扫描 PDF 需先 OCR。

## 模型与结果说明

- 实时匹配调用 `src.learning.workflow.match_documents` → 原 `MatchingPipeline`，使用词典技能和离线 hashed character n-grams，不下载模型。页面明确说明它不是 MiniLM / JobBERT 实验配置。
- 学习推荐调用 `LearningRecommender`，Hybrid 是 BM25 + TF-IDF 的 RRF 融合；主题覆盖不等于能力掌握。
- 实验页直接读取 `results/B_skill_extraction.csv`、`C_requirement_clf.csv`、`D_matching.csv`、`T_talentclef_ablation_minilm.csv`；不同面板的基准和指标不同。
- 分类标签来自单一 LLM 标注者，尚未人工复核；TalentCLEF dev 只有 10 个查询。
- 输入文档不写入项目目录；PDF/DOCX 解析临时文件随请求完成清理。仅监听 127.0.0.1。

## 验证

```powershell
python -m unittest discover -s tests -p "test_*.py"
node tests/showcase_ui_smoke.cjs
```

HTTP 测试使用项目已有的内存 socket 传输方式，验证实际请求解析、路由、上传、排序与推荐；不证明当前机器的端口可连接。JS 测试使用轻量 DOM 模型，验证内嵌案例与界面流程，不代替浏览器视觉检查。

如当前环境无法连接本地端口，可用 `Showcase.html` 完成内置案例展示。
