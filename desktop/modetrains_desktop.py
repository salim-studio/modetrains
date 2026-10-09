"""ModeTrains Desktop — graphical front-end for the modetrains library.

Runs on the standard library only (tkinter): no extra dependencies.
Subcommands are executed as ``python -m modetrains ...`` in a worker
thread with live log streaming, so the UI never freezes.

Usage:
    python modetrains_desktop.py              # launch the GUI
    python modetrains_desktop.py --self-test  # headless smoke test (CI)

Copyright (c) 2026 salim-slimani. MIT license.
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading

APP_NAME = "ModeTrains Desktop"
APP_VERSION = "0.2.0"
BRAND = "#4F46E5"
BRAND2 = "#06B6D4"
BG = "#F1F5F9"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# ---------------------------------------------------------------------------
# Headless-safe logic (importable and testable without a display)
# ---------------------------------------------------------------------------

def find_python() -> str:
    """Python interpreter used to run the modetrains CLI."""
    return sys.executable or "python"


def build_info_command(python: str | None = None) -> list[str]:
    return [python or find_python(), "-m", "modetrains", "info"]


def build_train_command(model: str, data: str, split: str = "train",
                        max_seq: int = 2048, steps: int = 60,
                        batch: int = 2, accum: int = 4,
                        out: str = "outputs",
                        python: str | None = None) -> list[str]:
    if not model.strip():
        raise ValueError("Model is required.")
    if not data.strip():
        raise ValueError("Dataset is required.")
    if max_seq < 256:
        raise ValueError("Max sequence length must be >= 256.")
    if steps < 1 or batch < 1 or accum < 1:
        raise ValueError("Steps / batch / accum must be >= 1.")
    return [python or find_python(), "-m", "modetrains", "train",
            "--model", model.strip(), "--data", data.strip(),
            "--split", split.strip() or "train", "--out", out.strip() or "outputs",
            "--max-seq", str(max_seq), "--steps", str(steps),
            "--batch", str(batch), "--accum", str(accum)]


def build_infer_command(model: str, prompt: str, tokens: int = 256,
                        full: bool = False,
                        python: str | None = None) -> list[str]:
    if not model.strip():
        raise ValueError("Model is required.")
    if not prompt.strip():
        raise ValueError("Prompt is required.")
    if tokens < 1 or tokens > 4096:
        raise ValueError("Tokens must be between 1 and 4096.")
    cmd = [python or find_python(), "-m", "modetrains", "infer",
           "--model", model.strip(), "--prompt", prompt.strip(),
           "--tokens", str(tokens)]
    if full:
        cmd.append("--full")
    return cmd


class JobRunner:
    """Runs one subprocess job with streaming output on a worker thread."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.running = False

    def start(self, cmd: list[str], cwd: str | None = None):
        if self.running:
            raise RuntimeError("A job is already running.")
        self.lines = queue.Queue()
        self.running = True
        self._thread = threading.Thread(target=self._run, args=(cmd, cwd), daemon=True)
        self._thread.start()

    def _run(self, cmd: list[str], cwd: str | None):
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=cwd)
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.lines.put(line.rstrip("\n"))
            self.proc.wait()
            self.lines.put(f"[exit code {self.proc.returncode}]")
        except Exception as e:  # e.g. interpreter missing
            self.lines.put(f"[error] {e}")
        finally:
            self.proc = None
            self.running = False
            self.lines.put(None)  # sentinel: stream finished

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def self_test() -> int:
    """Headless smoke test: no display needed. Returns process exit code."""
    checks = []
    checks.append(("version", APP_VERSION == "0.2.0"))
    c = build_train_command("m", "d")
    checks.append(("train-cmd", c[:3] == [find_python(), "-m", "modetrains"] and "--steps" in c))
    c2 = build_infer_command("m", "hello", tokens=32)
    checks.append(("infer-cmd", "--tokens" in c2 and "32" in c2))
    c3 = build_info_command()
    checks.append(("info-cmd", c3[-1] == "info"))
    try:
        import tkinter
        checks.append(("tkinter", True))
    except Exception:
        checks.append(("tkinter", False))
    try:
        build_train_command("", "d")
        checks.append(("validation", False))
    except ValueError:
        checks.append(("validation", True))
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} selftest:{name}")
        ok = ok and passed
    print("SELF-TEST " + ("OK " + APP_VERSION if ok else "FAILED"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# GUI (needs a display; only constructed in __main__ without --self-test)
# ---------------------------------------------------------------------------

def launch_gui():  # pragma: no cover — requires a display
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(f"{APP_NAME} v{APP_VERSION}")
            self.geometry("860x640")
            self.configure(bg=BG)
            self.runner = JobRunner()
            self._build()
            self._poll()
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        # -- layout ------------------------------------------------------
        def _build(self):
            header = tk.Frame(self, bg="#0B0E1A")
            header.pack(fill="x")
            tk.Label(header, text="mode", fg="white", bg="#0B0E1A",
                     font=("Segoe UI", 20, "bold")).pack(side="left", padx=(16, 0), pady=12)
            tk.Label(header, text="trains", fg=BRAND2, bg="#0B0E1A",
                     font=("Segoe UI", 20, "bold")).pack(side="left", pady=12)
            tk.Label(header, text=f"Desktop v{APP_VERSION}  ·  fast LLM fine-tuning",
                     fg="#CBD5E1", bg="#0B0E1A", font=("Segoe UI", 10)).pack(
                         side="right", padx=16)

            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("Accent.TButton", background=BRAND, foreground="white",
                            font=("Segoe UI", 10, "bold"), padding=8)
            style.map("Accent.TButton", background=[("active", BRAND2)])

            tabs = ttk.Notebook(self)
            tabs.pack(fill="both", expand=True, padx=12, pady=12)
            self.train_tab = ttk.Frame(tabs)
            self.infer_tab = ttk.Frame(tabs)
            self.info_tab = ttk.Frame(tabs)
            tabs.add(self.train_tab, text="  Train  ")
            tabs.add(self.infer_tab, text="  Infer  ")
            tabs.add(self.info_tab, text="  Hardware  ")
            self._build_train()
            self._build_infer()
            self._build_info()
            tk.Label(self, text="© 2026 salim-slimani · MIT",
                     bg=BG, fg="#64748B", font=("Segoe UI", 8)).pack(side="bottom", pady=4)

        def _field(self, parent, row, label, default="", width=52):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            var = tk.StringVar(value=default)
            ttk.Entry(parent, textvariable=var, width=width).grid(
                row=row, column=1, sticky="ew", padx=8, pady=4)
            parent.columnconfigure(1, weight=1)
            return var

        def _logbox(self, parent):
            box = scrolledtext.ScrolledText(parent, height=16, font=("Consolas", 9),
                                            bg="#0B0E1A", fg="#E2E8F0",
                                            insertbackground="white")
            box.pack(fill="both", expand=True, padx=8, pady=8)
            box.configure(state="disabled")
            return box

        # -- train tab ---------------------------------------------------
        def _build_train(self):
            form = ttk.Frame(self.train_tab)
            form.pack(fill="x", padx=4, pady=4)
            self.t_model = self._field(form, 0, "Model:", DEFAULT_MODEL)
            self.t_data = self._field(form, 1, "Dataset:", "yahma/alpaca-cleaned")
            row2 = ttk.Frame(form)
            row2.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
            self.t_steps = tk.StringVar(value="60")
            self.t_batch = tk.StringVar(value="2")
            self.t_accum = tk.StringVar(value="4")
            self.t_seq = tk.StringVar(value="2048")
            self.t_out = tk.StringVar(value="outputs")
            for i, (lbl, var, w) in enumerate([
                    ("Steps", self.t_steps, 6), ("Batch", self.t_batch, 5),
                    ("Accum", self.t_accum, 5), ("Max seq", self.t_seq, 7),
                    ("Out dir", self.t_out, 12)]):
                ttk.Label(row2, text=lbl + ":").pack(side="left", padx=(0 if i == 0 else 10, 2))
                ttk.Entry(row2, textvariable=var, width=w).pack(side="left")
            btns = ttk.Frame(self.train_tab)
            btns.pack(fill="x", padx=8, pady=4)
            self.train_btn = ttk.Button(btns, text="▶  Start training",
                                        style="Accent.TButton", command=self._start_train)
            self.train_btn.pack(side="left")
            ttk.Button(btns, text="■  Stop", command=self._stop_job).pack(side="left", padx=8)
            self.train_prog = ttk.Progressbar(btns, mode="indeterminate", length=180)
            self.train_prog.pack(side="right")
            self.train_log = self._logbox(self.train_tab)

        def _start_train(self):
            try:
                cmd = build_train_command(
                    self.t_model.get(), self.t_data.get(), steps=int(self.t_steps.get()),
                    batch=int(self.t_batch.get()), accum=int(self.t_accum.get()),
                    max_seq=int(self.t_seq.get()), out=self.t_out.get())
            except ValueError as e:
                messagebox.showerror(APP_NAME, str(e))
                return
            except Exception:
                messagebox.showerror(APP_NAME, "Steps / batch / accum / max seq must be integers.")
                return
            self._append(self.train_log, "$ " + " ".join(cmd) + "\n")
            try:
                self.runner.start(cmd)
            except RuntimeError as e:
                messagebox.showwarning(APP_NAME, str(e))
                return
            self.train_prog.start(12)

        # -- infer tab ---------------------------------------------------
        def _build_infer(self):
            form = ttk.Frame(self.infer_tab)
            form.pack(fill="x", padx=4, pady=4)
            self.i_model = self._field(form, 0, "Model:", "outputs")
            ttk.Label(form, text="Prompt:").grid(row=1, column=0, sticky="nw", padx=8, pady=4)
            self.i_prompt = scrolledtext.ScrolledText(form, height=4, font=("Segoe UI", 10))
            self.i_prompt.grid(row=1, column=1, sticky="ew", padx=8, pady=4)
            form.columnconfigure(1, weight=1)
            row = ttk.Frame(form)
            row.grid(row=2, column=1, sticky="w", padx=8, pady=4)
            ttk.Label(row, text="Max tokens:").pack(side="left")
            self.i_tokens = tk.StringVar(value="256")
            ttk.Entry(row, textvariable=self.i_tokens, width=7).pack(side="left", padx=6)
            self.i_full = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text="Full precision (no 4-bit)", variable=self.i_full).pack(side="left", padx=10)
            btns = ttk.Frame(self.infer_tab)
            btns.pack(fill="x", padx=8, pady=4)
            ttk.Button(btns, text="✨  Generate", style="Accent.TButton",
                       command=self._start_infer).pack(side="left")
            ttk.Button(btns, text="■  Stop", command=self._stop_job).pack(side="left", padx=8)
            self.infer_log = self._logbox(self.infer_tab)

        def _start_infer(self):
            try:
                tokens = int(self.i_tokens.get())
                cmd = build_infer_command(self.i_model.get(),
                                          self.i_prompt.get("1.0", "end"),
                                          tokens=tokens, full=self.i_full.get())
            except ValueError as e:
                messagebox.showerror(APP_NAME, str(e))
                return
            except Exception:
                messagebox.showerror(APP_NAME, "Max tokens must be an integer.")
                return
            self._append(self.infer_log, "$ " + " ".join(cmd) + "\n")
            try:
                self.runner.start(cmd)
            except RuntimeError as e:
                messagebox.showwarning(APP_NAME, str(e))

        # -- info tab ----------------------------------------------------
        def _build_info(self):
            bar = ttk.Frame(self.info_tab)
            bar.pack(fill="x", padx=8, pady=8)
            ttk.Button(bar, text="⟳  Refresh hardware info", style="Accent.TButton",
                       command=self._refresh_info).pack(side="left")
            self.info_log = self._logbox(self.info_tab)
            self._refresh_info()

        def _refresh_info(self):
            self._append(self.info_log, "$ " + " ".join(build_info_command()) + "\n")
            try:
                out = subprocess.run(build_info_command(), capture_output=True,
                                     text=True, timeout=120).stdout or "(no output)"
            except Exception as e:
                out = f"[error] {e}"
            try:
                out = json.dumps(json.loads(out.split("}\n{")[0] + "}"), indent=2)
            except Exception:
                pass
            self._append(self.info_log, out + "\n")

        # -- shared ------------------------------------------------------
        def _append(self, box, text):
            box.configure(state="normal")
            box.insert("end", text)
            box.see("end")
            box.configure(state="disabled")

        def _stop_job(self):
            self.runner.stop()

        def _poll(self):
            target = self.train_log if self._job_is_train() else self.infer_log
            try:
                while True:
                    line = self.runner.lines.get_nowait()
                    if line is None:
                        self.train_prog.stop()
                        break
                    self._append(target, line + "\n")
            except queue.Empty:
                pass
            self.after(120, self._poll)

        def _job_is_train(self):
            try:
                return self.runner.proc is not None and "train" in (self.runner.proc.args or [])
            except Exception:
                return True

        def _on_close(self):
            self._stop_job()
            self.destroy()

    App().mainloop()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    launch_gui()
