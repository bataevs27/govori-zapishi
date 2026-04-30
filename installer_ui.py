#!/usr/bin/env python3
"""Govori-Zapishi — окно прогресса установки (запускается лаунчером)"""
import tkinter as tk
from tkinter import ttk
import os
import time
import threading

PROGRESS_FILE = "/tmp/gz_install_progress"


class InstallerWindow:
    def __init__(self, root):
        self.root = root
        root.title("Govori-Zapishi")
        root.resizable(False, False)
        root.geometry("440x140")
        root.eval("tk::PlaceWindow . center")
        root.lift()
        ttk.Style().theme_use("aqua")

        try:
            import AppKit
            AppKit.NSApplication.sharedApplication().setActivationPolicy_(
                AppKit.NSApplicationActivationPolicyAccessory)
        except Exception:
            pass

        frame = ttk.Frame(root, padding="24 20 24 20")
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Установка зависимостей",
                  font=("System", 14, "bold")).pack(anchor="w")

        self.step_var = tk.StringVar(value="Подготовка...")
        ttk.Label(frame, textvariable=self.step_var,
                  foreground="gray").pack(anchor="w", pady=(4, 10))

        self.bar = ttk.Progressbar(frame, mode="determinate", length=392)
        self.bar.pack(fill="x")

        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while True:
            try:
                if os.path.exists(PROGRESS_FILE):
                    line = open(PROGRESS_FILE).read().strip()
                    if line == "done":
                        self.root.after(0, self.root.destroy)
                        return
                    if "|" in line:
                        name, cur, total = line.split("|")
                        cur, total = int(cur), int(total)
                        pct = int(cur / total * 100)
                        self.root.after(0, self._update, name, cur, total, pct)
            except Exception:
                pass
            time.sleep(0.5)

    def _update(self, name, cur, total, pct):
        self.step_var.set(f"{name}   ({cur} / {total})")
        self.bar["value"] = pct


if __name__ == "__main__":
    root = tk.Tk()
    InstallerWindow(root)
    root.mainloop()
