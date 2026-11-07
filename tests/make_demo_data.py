"""
Generate a SMALL SYNTHETIC demo set used only to check that every module runs.
!! These are NOT real data and the labels are NOT human annotations — do not report results on them. !!
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data" / "demo"
ROOT.mkdir(parents=True, exist_ok=True)

# (section header, sentence, gold label)
JDS = {
 "JD_data_analyst": [
  ("About us", "We are a fast-growing e-commerce company in Singapore.", "UNSPECIFIED"),
  ("Responsibilities", "Build dashboards in Power BI and Tableau to track sales performance.", "UNSPECIFIED"),
  ("Responsibilities", "Write SQL queries to extract and clean data from PostgreSQL.", "UNSPECIFIED"),
  ("Responsibilities", "Present insights to business stakeholders every week.", "UNSPECIFIED"),
  ("Requirements", "Bachelor's degree in Statistics, Mathematics, Computer Science or a related field.", "REQUIRED"),
  ("Requirements", "At least 2 years of experience in data analysis.", "REQUIRED"),
  ("Requirements", "Proficient in SQL and Excel.", "REQUIRED"),
  ("Requirements", "Experience with Python (Pandas, NumPy) for data analysis.", "REQUIRED"),
  ("Requirements", "Strong communication skills.", "REQUIRED"),
  ("Nice to have", "Knowledge of A/B testing and hypothesis testing.", "PREFERRED"),
  ("Nice to have", "Experience with Google Analytics.", "PREFERRED"),
  ("Nice to have", "Familiarity with dbt or Airflow.", "PREFERRED"),
 ],
 "JD_backend_engineer": [
  ("The role", "You will design and build REST APIs for our payments platform.", "UNSPECIFIED"),
  ("The role", "You will maintain microservices running on Kubernetes in AWS.", "UNSPECIFIED"),
  ("What you need", "Minimum 3 years of professional experience in backend development.", "REQUIRED"),
  ("What you need", "Strong knowledge of Java or Go.", "REQUIRED"),
  ("What you need", "Experience with PostgreSQL and Redis.", "REQUIRED"),
  ("What you need", "Familiarity with Docker, CI/CD and Git.", "REQUIRED"),
  ("What you need", "Applicants must be Singapore Citizens or Permanent Residents; others will not be considered.", "MANDATORY"),
  ("Bonus points", "Experience with Kafka.", "PREFERRED"),
  ("Bonus points", "Contributions to open-source projects would be a plus.", "PREFERRED"),
  ("Benefits", "Flexible working hours and an annual learning budget.", "UNSPECIFIED"),
 ],
 "JD_ml_engineer": [
  ("Overview", "Join our applied AI team building NLP products for healthcare.", "UNSPECIFIED"),
  ("Responsibilities", "Fine-tune large language models with PyTorch and Hugging Face Transformers.", "UNSPECIFIED"),
  ("Responsibilities", "Deploy models as APIs using FastAPI and Docker.", "UNSPECIFIED"),
  ("Responsibilities", "Track experiments with MLflow and evaluate model quality.", "UNSPECIFIED"),
  ("Qualifications", "Master's degree or PhD in Computer Science, Machine Learning or a related field.", "REQUIRED"),
  ("Qualifications", "3+ years of experience in machine learning engineering.", "REQUIRED"),
  ("Qualifications", "Proficiency in Python and deep learning frameworks such as PyTorch or TensorFlow.", "REQUIRED"),
  ("Qualifications", "Solid understanding of natural language processing.", "REQUIRED"),
  ("Preferred qualifications", "Experience with retrieval-augmented generation and LangChain.", "PREFERRED"),
  ("Preferred qualifications", "Publications at NLP conferences are desirable.", "PREFERRED"),
  ("Preferred qualifications", "Experience with Kubernetes and Google Cloud.", "PREFERRED"),
 ],
}

GOLD_SKILLS = {
 "JD_data_analyst": ["Power BI", "Tableau", "SQL", "PostgreSQL", "Excel", "Python", "Pandas", "NumPy",
                     "Data analysis", "Communication", "A/B testing", "Hypothesis testing", "Google Analytics",
                     "dbt", "Airflow"],
 "JD_backend_engineer": ["REST API", "Microservices", "Kubernetes", "AWS", "Java", "Go", "PostgreSQL", "Redis",
                         "Docker", "CI/CD", "Git", "Kafka"],
 "JD_ml_engineer": ["Natural language processing", "Large language models", "Fine-tuning", "PyTorch",
                    "Hugging Face", "Transformers", "APIs", "FastAPI", "Docker", "MLflow", "Model evaluation",
                    "Python", "Deep learning", "TensorFlow", "Retrieval-augmented generation", "LangChain",
                    "Kubernetes", "Google Cloud", "Machine learning"],
}

RESUMES = {
 "R01_analyst_strong": """Summary
Data analyst with 3 years of experience turning data into business insights.
Experience
Data Analyst
Shopee | Singapore
Jan 2022 - Present
Built Power BI dashboards tracking GMV and conversion for 5 business units.
Wrote complex SQL queries on PostgreSQL to clean and aggregate sales data.
Designed A/B tests and ran hypothesis testing to evaluate promotions.
Education
Bachelor of Science in Statistics
National University of Singapore
Aug 2017 - May 2021
Skills
SQL, Python (Pandas, Num
Py), Excel, Power BI, Tableau, Google Analytics""",
 "R02_analyst_junior": """Education
Diploma in Business Analytics
Ngee Ann Polytechnic
Apr 2020 - Mar 2023
Experience
Business Analyst Intern
DBS Bank | Singapore
May 2023 - Dec 2023
Prepared weekly Excel reports and pivot tables for the operations team.
Skills
Excel, PowerPoint, basic SQL""",
 "R03_backend_senior": """Profile
Backend engineer with 5 years building payment systems.
Work Experience
Senior Software Engineer
Grab | Singapore
Mar 2020 - Present
Designed REST APIs in Go and Java serving 10k requests per second.
Operated microservices on Kubernetes (AWS EKS) with Docker and CI/CD pipelines in GitLab.
Used PostgreSQL, Redis and Kafka for event-driven processing.
Education
Bachelor of Engineering in Computer Engineering
Nanyang Technological University
Aug 2015 - May 2019
Skills
Go, Java, PostgreSQL, Redis, Kafka, Docker, Kubernetes, AWS, Git""",
 "R04_frontend_dev": """Experience
Frontend Developer
Carousell | Singapore
Jun 2021 - Present
Built React and TypeScript single page applications.
Improved web performance by 30% using code splitting.
Education
Bachelor of Computing in Computer Science
National University of Singapore
Aug 2017 - May 2021
Skills
JavaScript, TypeScript, React, HTML, CSS, Node.js, Git""",
 "R05_ml_phd": """Summary
Machine learning researcher focusing on natural language processing and large language models.
Experience
Research Engineer
AI Singapore | Singapore
Jul 2021 - Present
Fine-tuned LLMs with PyTorch, Hugging Face Transformers and DeepSpeed on GPU clusters.
Built a retrieval-augmented generation system with LangChain for document QA.
Deployed models with FastAPI and Docker; tracked experiments in MLflow.
Education
PhD in Computer Science
Nanyang Technological University
Aug 2017 - Jun 2021
Skills
Python, PyTorch, TensorFlow, Transformers, LangChain, Docker, Kubernetes, Google Cloud""",
 "R06_ml_junior": """Education
Bachelor of Engineering with Honours in Computer Engineering
Singapore Institute of Technology
Aug 2021 - Jul 2025
Projects
Sentiment classifier for product reviews using scikit-learn and a fine-tuned BERT model.
Experience
AI Intern
Startup XYZ | Singapore
Jan 2024 - Aug 2024
Trained text classification models in PyTorch and wrote evaluation scripts.
Skills
Python, PyTorch, Scikit-learn, NLP, Git""",
 "R07_mech_engineer": """Experience
Mechanical Engineer
Keppel | Singapore
Feb 2018 - Present
Designed offshore structures using SolidWorks and ANSYS.
Managed vendor quality inspections.
Education
Bachelor of Engineering in Mechanical Engineering
National University of Singapore
Aug 2013 - May 2017
Skills
SolidWorks, ANSYS, AutoCAD, Project management""",
 "R08_data_scientist": """Summary
Data scientist with experience in machine learning and analytics.
Experience
Data Scientist
Lazada | Singapore
Aug 2020 - Present
Built churn prediction models with XGBoost and Scikit-learn.
Analysed customer data with SQL and Python (Pandas) and presented insights to stakeholders.
Education
Master of Science in Data Science
Singapore Management University
Aug 2019 - Jul 2020
Skills
Python, SQL, Pandas, Scikit-learn, XGBoost, Tableau, Airflow""",
}

# (jd, resume) -> label 0/1/2   (author's judgement, demo only)
PAIRS = {
 ("JD_data_analyst", "R01_analyst_strong"): 2, ("JD_data_analyst", "R02_analyst_junior"): 1,
 ("JD_data_analyst", "R03_backend_senior"): 0, ("JD_data_analyst", "R04_frontend_dev"): 0,
 ("JD_data_analyst", "R05_ml_phd"): 1, ("JD_data_analyst", "R06_ml_junior"): 0,
 ("JD_data_analyst", "R07_mech_engineer"): 0, ("JD_data_analyst", "R08_data_scientist"): 2,
 ("JD_backend_engineer", "R01_analyst_strong"): 0, ("JD_backend_engineer", "R02_analyst_junior"): 0,
 ("JD_backend_engineer", "R03_backend_senior"): 2, ("JD_backend_engineer", "R04_frontend_dev"): 1,
 ("JD_backend_engineer", "R05_ml_phd"): 1, ("JD_backend_engineer", "R06_ml_junior"): 0,
 ("JD_backend_engineer", "R07_mech_engineer"): 0, ("JD_backend_engineer", "R08_data_scientist"): 0,
 ("JD_ml_engineer", "R01_analyst_strong"): 0, ("JD_ml_engineer", "R02_analyst_junior"): 0,
 ("JD_ml_engineer", "R03_backend_senior"): 0, ("JD_ml_engineer", "R04_frontend_dev"): 0,
 ("JD_ml_engineer", "R05_ml_phd"): 2, ("JD_ml_engineer", "R06_ml_junior"): 1,
 ("JD_ml_engineer", "R07_mech_engineer"): 0, ("JD_ml_engineer", "R08_data_scientist"): 1,
}


def jd_text(rows):
    out, cur = [], None
    for hdr, sent, _ in rows:
        if hdr != cur:
            out.append(f"\n{hdr}:")
            cur = hdr
        out.append(f"• {sent}")
    return "\n".join(out).strip()


def main():
    (ROOT / "jds").mkdir(exist_ok=True)
    (ROOT / "resumes").mkdir(exist_ok=True)
    for jid, rows in JDS.items():
        (ROOT / "jds" / f"{jid}.txt").write_text(jd_text(rows), encoding="utf-8")
    for rid, t in RESUMES.items():
        (ROOT / "resumes" / f"{rid}.txt").write_text(t, encoding="utf-8")
    pd.DataFrame([{"jd_id": j, "section_header": h, "sentence": s, "gold_label": l}
                  for j, rows in JDS.items() for h, s, l in rows]).to_csv(ROOT / "demo_requirement_labels.csv", index=False)
    pd.DataFrame([{"doc_id": d, "skill": s, "keep(1/0)": 1} for d, sk in GOLD_SKILLS.items() for s in sk]) \
        .to_csv(ROOT / "demo_skill_gold.csv", index=False)
    pd.DataFrame([{"jd_id": j, "resume_id": r, "label": l} for (j, r), l in PAIRS.items()]) \
        .to_csv(ROOT / "demo_match_labels.csv", index=False)
    print("demo data written to", ROOT)


if __name__ == "__main__":
    main()
