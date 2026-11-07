# 标注规范 Annotation Guidelines（Group 32）

所有人标注前先读一遍；正式标注前，每人先试标同一份 **5 个 JD + 10 对 JD–简历**，算一致性（κ），讨论分歧后再修改本规范。
把讨论后的修改记录在本文件末尾的 "Changelog"，报告里可以直接引用。

---

## 任务 1：JD 句子优先级（`requirements_to_label.csv` → 填 `gold_label`）

| 标签 | 定义 | 典型线索 | 例子 |
|---|---|---|---|
| **MANDATORY** | 硬性资格，不满足会被直接拒绝 | "only … will be considered", "must be a Singapore Citizen", "not eligible" | Applicants must be Singapore Citizens or PRs. |
| **REQUIRED** | 职位要求的能力/经验/学历（不满足会明显扣分，但不是一票否决） | must, required, at least, proficient in, 位于 Requirements / Qualifications 段落下 | At least 2 years of experience in SQL. |
| **PREFERRED** | 加分项 | preferred, a plus, nice to have, advantageous, desirable, 位于 Nice to have 段落下 | Experience with Kafka is a plus. |
| **UNSPECIFIED** | 不是对候选人的要求：公司介绍、职责描述、福利、申请说明 | Responsibilities / About us / Benefits | You will design REST APIs. |

规则：
1. **句子里的明确措辞优先于段落标题**：在 "Requirements:" 下面出现 "…would be advantageous" → PREFERRED。
2. 职责描述（"You will build…"）一律 UNSPECIFIED，即使提到了技能。
3. 一句话里混有必需和加分（"Python is required; Go is a plus"）→ 按**整句主要意思**标，并在 `notes` 里写 "mixed"。
4. `rule_suggestion` 列只是系统的建议，**不要照抄**，要独立判断。

## 任务 2：技能清单（`jd_skills_to_label.csv` / `resume_skills_to_label.csv`）

- 技能 = 可以被检验的能力：编程语言、库/框架、工具/平台、技术方法（fine-tuning, A/B testing）、领域知识（NLP, statistics），以及**明确写出的**软技能（communication）。
- **不算技能**：职责句子（"build scalable pipelines"）、性格形容词（curious, resilient）、公司名、学位名称。
- 写**简短规范名**：`PyTorch`、`CI/CD`、`instruction tuning`；枚举要拆开（"Linux, Docker and CI/CD" → 3 行）。
- 系统预填的建议：正确的在 `keep(1/0)` 填 1，错误的填 0；漏掉的**新增一行**并填 1。

## 任务 3：JD–简历匹配（`pairs_to_label.csv`）

只看 JD 和简历内容，**不看姓名、性别、学校声誉**（标注前用 `du.anonymise_resume` 去掉联系方式和姓名）。

| label | 含义 | 判断参考 |
|---|---|---|
| **2 = Good fit** | 值得进面试 | 满足几乎所有 REQUIRED，学历/年限达到或接近 |
| **1 = Potential fit** | 可以考虑 | 满足一半左右 REQUIRED，或技能相关但年限/学历不足 |
| **0 = No fit** | 不合适 | 领域明显不同，或缺少大部分核心要求 |

另外填 `score_0_100`（主观分，用于 Spearman）和一句话 `rationale`（错误分析时非常有用）。

## 一致性与裁决

- 至少 20% 的样本由 2 人以上独立标注，用 `du.agreement_report` 计算 Cohen's κ / Fleiss' κ。
  κ > 0.6 可以接受；若 < 0.6，讨论分歧、修改规范后重标。
- 最终标签用 `du.adjudicate`（多数投票）；平票（`tie=True`）由第三人裁决。

## Changelog
- v1 (初版)
- v2 (after labelling 30 JDs from the HF resume–JD fit dataset, 749 sentences):
  * "The ideal candidate has/will have …" → PREFERRED when the JD has a separate required-qualifications list,
    otherwise REQUIRED; "X years is ideal" → PREFERRED.
  * Physical demands, EEO statements, salary/benefits, agency notices, application instructions → UNSPECIFIED.
  * Pre-employment conditions that block hiring (drug test after conditional offer, work authorisation,
    clearance) → MANDATORY.
  * Glued sentences (scraping artefact) are labelled by their requirement part and marked `mixed` in notes.
