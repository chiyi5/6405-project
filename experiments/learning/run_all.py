"""Reproduce every learning-module result in order.

    python experiments/learning/run_all.py             # CPU part (~6 min): benchmark build, L1, L2, L3
    python experiments/learning/run_all.py --neural    # + L4 fine-tuning and Sentence-BERT / cross-encoder systems

Each step is a separate process so one failure is reported without hiding the others.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "experiments" / "learning"
neural = "--neural" in sys.argv
steps = [[sys.executable, "-c", "import sys; sys.path.insert(0, '.'); from src.learning.benchmark import save_benchmark; print(save_benchmark())"]]
if neural:
    steps.append([sys.executable, str(E / "L4_finetune_biencoder.py")])
    steps.append([sys.executable, str(E / "L1_retrieval_benchmark.py"), "--neural", "--finetuned", str(ROOT / "models" / "course_biencoder")])
    steps.append([sys.executable, str(E / "L2_plan_evaluation.py"), "--neural"])
else:
    steps.append([sys.executable, str(E / "L1_retrieval_benchmark.py")])
    steps.append([sys.executable, str(E / "L2_plan_evaluation.py")])
steps.append([sys.executable, str(E / "L3_error_analysis.py")])
if not (ROOT / "data" / "learning" / "human" / "sheet_A.csv").exists():
    steps.append([sys.executable, str(E / "L5_human_judgements.py"), "make"])

failed = []
for cmd in steps:
    print("\n=== " + " ".join(Path(c).name for c in cmd[1:]), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode:
        failed.append(Path(cmd[1]).name)
print("\nFAILED: " + ", ".join(failed) if failed else "\nall steps finished; results in results/learning/bench/")
sys.exit(1 if failed else 0)
