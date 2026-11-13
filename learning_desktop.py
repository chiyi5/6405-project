"""Native desktop entry point for environments where loopback HTTP is blocked.

Uses exactly the same Python matcher/recommender as the web demo. No sockets.
"""
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.learning import LearningRecommender, gaps_from_match, parse_gaps
from src.learning.workflow import match_documents, plan_markdown


class LearningDesktop:
    def __init__(self, root):
        self.root = root
        self.engine = LearningRecommender()
        self.events = queue.Queue()
        self.plan = None
        self.match = None
        root.title("SkillBridge · Group 32 学习路径推荐")
        root.geometry("1220x820")
        root.minsize(900, 650)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=8)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"))
        shell = ttk.Frame(root, padding=18)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="SkillBridge · 技能缺口 → 学习路径", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, text="本地桌面模式 · 无需浏览器或端口 · 复用小组 MatchingPipeline 与独立推荐模块").pack(anchor="w", pady=(4, 12))
        inputs = ttk.Notebook(shell)
        inputs.pack(fill="x")
        docs = ttk.Frame(inputs, padding=10)
        manual = ttk.Frame(inputs, padding=10)
        inputs.add(docs, text="从 JD 和简历开始")
        inputs.add(manual, text="手动技能／导入匹配结果")
        docs.columnconfigure((0, 1), weight=1)
        ttk.Label(docs, text="英文职位描述 JD").grid(row=0, column=0, sticky="w")
        ttk.Label(docs, text="英文简历 Resume").grid(row=0, column=1, sticky="w")
        self.jd = self.textbox(docs, "Requirements:\nPython, SQL, Docker, FastAPI, Git.\nPreferred: Testing and machine learning.", 6)
        self.cv = self.textbox(docs, "Skills:\nPython\nProjects:\nBuilt a Python script for processing text files.", 6)
        self.jd.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.cv.grid(row=1, column=1, sticky="ew")
        self.buttons = []
        self.button(docs, "匹配文档并推荐", lambda: self.run("documents")).grid(row=2, column=0, sticky="w", pady=(8, 0))
        manual.columnconfigure(0, weight=1)
        ttk.Label(manual, text="每行一个技能，可加 | REQUIRED / PREFERRED / MANDATORY").grid(row=0, column=0, sticky="w")
        self.gaps = self.textbox(manual, "SQL | REQUIRED\nDocker | REQUIRED\nFastAPI | PREFERRED", 5)
        self.gaps.grid(row=1, column=0, sticky="ew")
        actions = ttk.Frame(manual)
        actions.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.button(actions, "按手动技能推荐", lambda: self.run("manual")).pack(side="left")
        self.button(actions, "导入小组匹配 JSON", self.import_match).pack(side="left", padx=8)
        controls = ttk.Frame(shell)
        controls.pack(fill="x", pady=12)
        ttk.Label(controls, text="推荐方法").pack(side="left")
        self.method = tk.StringVar(value="hybrid")
        ttk.Combobox(controls, textvariable=self.method, values=["bm25", "tfidf", "hybrid"], state="readonly", width=12).pack(side="left", padx=8)
        self.optimize = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="覆盖与去重优化", variable=self.optimize).pack(side="left", padx=8)
        ttk.Label(controls, text="最多推荐").pack(side="left", padx=(12, 4))
        self.top_k = tk.StringVar(value="5")
        ttk.Spinbox(controls, from_=1, to=12, textvariable=self.top_k, width=4).pack(side="left")
        self.button(controls, "对比四组算法", lambda: self.run("compare")).pack(side="left", padx=12)
        self.button(controls, "导出 JSON", lambda: self.export("json")).pack(side="right")
        self.button(controls, "导出 Markdown", lambda: self.export("md")).pack(side="right", padx=8)
        self.status = tk.StringVar(value="示例输入为合成数据。请选择输入方式并运行。")
        ttk.Label(shell, textvariable=self.status, wraplength=1150).pack(anchor="w", pady=(0, 8))
        outputs = ttk.Notebook(shell)
        outputs.pack(fill="both", expand=True)
        self.outputs = {}
        for key, title in [("plan", "学习路径与来源"), ("match", "小组匹配结果"), ("compare", "算法对比"), ("trace", "评分分解")]:
            frame = ttk.Frame(outputs)
            outputs.add(frame, text=title)
            text = self.textbox(frame, "", 10)
            scroll = ttk.Scrollbar(frame, command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            text.pack(fill="both", expand=True)
            self.outputs[key] = text
        ttk.Label(shell, text="12 个学习单元的起步语料；主题覆盖 ≠ 掌握技能。正式 NDCG／Recall／MRR 仍需人工标注。", wraplength=1150).pack(anchor="w", pady=(10, 0))
        root.after(100, self.poll)

    @staticmethod
    def textbox(parent, value, height):
        text = tk.Text(parent, height=height, width=40, wrap="word", font=("Microsoft YaHei UI", 10), padx=10, pady=8)
        text.insert("1.0", value)
        return text

    def button(self, parent, title, command):
        button = ttk.Button(parent, text=title, command=command)
        self.buttons.append(button)
        return button

    def show(self, key, value):
        self.outputs[key].delete("1.0", "end")
        self.outputs[key].insert("1.0", value)

    def run(self, mode, imported=None):
        try:
            options = {"method": self.method.get(), "top_k": int(self.top_k.get()), "optimize": self.optimize.get()}
            # Read Tk state only on the main UI thread.
            jd, cv = self.jd.get("1.0", "end"), self.cv.get("1.0", "end")
            manual = self.gaps.get("1.0", "end")
            match = imported if imported is not None else self.match
            if mode == "compare" and self.plan is None:
                raise ValueError("请先匹配文档或生成一次推荐，再比较同一组技能缺口。")
            previous_gaps = self.plan["gaps"] if self.plan else None
        except Exception as exc:
            messagebox.showerror("输入有误", str(exc))
            return
        for button in self.buttons:
            button.state(["disabled"])
        self.status.set("正在本地计算…")

        def work():
            try:
                if mode == "documents":
                    current_match = match_documents(jd, cv)
                    gaps = gaps_from_match(current_match)
                elif mode == "import":
                    current_match, gaps = match, gaps_from_match(match)
                else:
                    current_match = None
                    gaps = previous_gaps if mode == "compare" else parse_gaps(manual)
                if mode == "compare":
                    results = [self.engine.recommend(gaps, method=m, optimize=o, top_k=options["top_k"])
                               for m, o in [("bm25", False), ("tfidf", False), ("hybrid", False), ("hybrid", True)]]
                    self.events.put(("compare", results))
                else:
                    self.events.put(("result", (self.engine.recommend(gaps, **options), current_match)))
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            kind, data = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            for button in self.buttons:
                button.state(["!disabled"])
            if kind == "error":
                self.status.set("运行失败：" + data)
                messagebox.showerror("运行失败", data)
            elif kind == "compare":
                rows = ["当前输入的四组消融对比（不是独立人工测试集的准确率）\n"]
                for r in data:
                    rows.append(f"{r['backend']} | 优化={r['optimized']}\n主题覆盖={r['weighted_coverage']:.1%}  重复率={r['redundancy_ratio']:.1%}  单元数={len(r['recommendations'])}  单次耗时={r['latency_ms']} ms\n")
                self.show("compare", "\n".join(rows))
                self.status.set("四组对比完成，请切换到“算法对比”页签。")
            else:
                self.plan, self.match = data
                self.show("plan", plan_markdown(self.plan))
                self.show("trace", json.dumps(self.plan["selection_trace"], ensure_ascii=False, indent=2))
                self.show("match", json.dumps(self.match, ensure_ascii=False, indent=2) if self.match else "当前使用手动技能输入。")
                self.show("compare", "输入已更新，请重新点击“对比四组算法”。")
                self.status.set(f"完成：{len(self.plan['gaps'])} 个技能缺口，{len(self.plan['recommendations'])} 个推荐单元，加权主题覆盖 {self.plan['weighted_coverage']:.1%}。")
        self.root.after(100, self.poll)

    def import_match(self):
        path = filedialog.askopenfilename(filetypes=[("匹配结果 JSON", "*.json")])
        if path:
            try:
                result = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                self.run("import", result)
            except Exception as exc:
                messagebox.showerror("导入失败", str(exc))

    def export(self, extension):
        if self.plan is None:
            messagebox.showinfo("尚无结果", "请先生成一次学习路径。")
            return
        path = filedialog.asksaveasfilename(defaultextension="." + extension, initialfile="learning_plan." + extension)
        if path:
            content = json.dumps(self.plan, ensure_ascii=False, indent=2) if extension == "json" else plan_markdown(self.plan)
            Path(path).write_text(content, encoding="utf-8")
            self.status.set("已导出：" + path)


if __name__ == "__main__":
    root = tk.Tk()
    LearningDesktop(root)
    root.mainloop()
