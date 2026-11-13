"""Execute only notebook section 9 and save its actual outputs, without Jupyter server."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    notebook_path = ROOT / "Group32_NLPproject_extended.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or not cell.get("id", "").startswith("learning-"):
            continue
        count += 1
        stream = io.StringIO()
        outputs = []

        def flush():
            text = stream.getvalue()
            if text:
                outputs.append({"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)})
                stream.seek(0)
                stream.truncate(0)

        def display(*objects, **_):
            flush()
            for obj in objects:
                data = {"text/plain": str(obj).splitlines(keepends=True)}
                if hasattr(obj, "to_html"):
                    data["text/html"] = obj.to_html(index=False).splitlines(keepends=True)
                outputs.append({"output_type": "display_data", "data": data, "metadata": {}})

        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), patch("IPython.display.display", display):
            exec(compile("".join(cell["source"]), f"{notebook_path.name}:{cell['id']}", "exec"), namespace)
        flush()
        cell["outputs"] = outputs
        cell["execution_count"] = count
        print(f"PASS {cell['id']}")
    if count != 4:
        raise RuntimeError(f"Expected 4 executable extension cells, got {count}")
    # Write only once all cells finish successfully; preserve all original cell content.
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(namespace["learning_comparison"].to_string(index=False))
    print("Uncovered topics:", ", ".join(namespace["learning_plan"]["uncovered_skills"]))
    print("Saved actual outputs in notebook section 9 and results/learning/L_*")


if __name__ == "__main__":
    main()
