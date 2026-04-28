#!/usr/bin/env python3
"""Stenograf — окно настроек (запускается как отдельный процесс)"""
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import os
import json
import subprocess
import requests

CONFIG_FILE = os.path.expanduser("~/.stenograf_config.json")
TOKEN_FILE  = os.path.expanduser("~/.stenograf_token")
HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
HF_LICENSES = [
    ("Диаризация спикеров",   "pyannote/speaker-diarization-3.1"),
    ("Сегментация аудио",     "pyannote/segmentation-3.0"),
    ("Сообщество диаризации", "pyannote/speaker-diarization-community-1"),
]


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f)

def load_token():
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE).read().strip()
    return ""

def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token.strip())

def link_label(parent, text, url=None, command=None, **kwargs):
    """Кликабельный label без выделения текста."""
    lbl = tk.Label(parent, text=text, foreground="#007AFF",
                   cursor="pointinghand", highlightthickness=0,
                   bd=0, **kwargs)
    action = command or (lambda e: subprocess.Popen(["open", url]))
    lbl.bind("<Button-1>", action)
    # Не даём тексту выделяться при клике
    lbl.bind("<ButtonRelease-1>", lambda e: lbl.selection_clear() if hasattr(lbl, 'selection_clear') else None)
    return lbl

def add_paste_menu(entry):
    """Правый клик → контекстное меню с Copy/Paste для Entry."""
    menu = tk.Menu(entry, tearoff=0)
    menu.add_command(label="Вырезать",  command=lambda: entry.event_generate("<<Cut>>"))
    menu.add_command(label="Копировать",command=lambda: entry.event_generate("<<Copy>>"))
    menu.add_command(label="Вставить",  command=lambda: entry.event_generate("<<Paste>>"))
    menu.add_separator()
    menu.add_command(label="Выделить всё", command=lambda: entry.select_range(0, "end"))

    def show(event):
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    entry.bind("<Button-2>", show)
    entry.bind("<Button-3>", show)


class SettingsApp:
    def __init__(self, root):
        self.root = root
        root.title("Stenograf — Настройки")
        root.resizable(False, False)
        root.lift()
        root.attributes("-topmost", True)
        root.after(200, lambda: root.attributes("-topmost", False))
        ttk.Style().theme_use("aqua")
        self._build()

    def _build(self):
        cfg   = load_config()
        token = load_token()

        outer = ttk.Frame(self.root, padding="20 16 20 16")
        outer.pack(fill="both", expand=True)

        def btn(parent, text, command, **kw):
            """Кнопка без focus-ring после клика."""
            b = ttk.Button(parent, text=text, command=command, takefocus=0, **kw)
            return b

        # ── Папки ─────────────────────────────────────────────────────────
        ttk.Label(outer, text="📁  Папки", font=("System", 13, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.meeting_var = tk.StringVar(value=cfg.get("output_dir", "не выбрана"))
        self.note_var    = tk.StringVar(value=cfg.get("note_dir",   "не выбрана"))

        for r, label_text, var, command in [
            (1, "Встречи:", self.meeting_var, self._change_meeting),
            (2, "Заметки:", self.note_var,    self._change_note),
        ]:
            ttk.Label(outer, text=label_text, width=9).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Label(outer, textvariable=var, foreground="gray",
                      width=42, anchor="w").grid(row=r, column=1, sticky="w", padx=6)
            btn(outer, "Изменить", command).grid(row=r, column=2, sticky="e")

        ttk.Separator(outer).grid(row=3, column=0, columnspan=3, sticky="ew", pady=12)

        # ── Токен ─────────────────────────────────────────────────────────
        ttk.Label(outer, text="🔑  HuggingFace токен", font=("System", 13, "bold")).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(0, 4))

        info = ttk.Frame(outer)
        info.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        ttk.Label(info, text="Зарегистрируйтесь и создайте Read-токен:", foreground="gray").pack(side="left")
        link_label(info, "Получить токен →", url=HF_TOKEN_URL).pack(side="right")

        self.token_var = tk.StringVar(value=token)
        token_entry = ttk.Entry(outer, textvariable=self.token_var, width=55)
        token_entry.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        add_paste_menu(token_entry)
        # На macOS Command = Meta в tkinter
        for key, event in [("v", "<<Paste>>"), ("c", "<<Copy>>"), ("x", "<<Cut>>")]:
            token_entry.bind(f"<Meta-{key}>", lambda e, ev=event: (
                token_entry.event_generate(ev), "break")[1])
        token_entry.bind("<Meta-a>", lambda e: (
            token_entry.select_range(0, "end"), "break")[1])
        token_entry.focus_set()

        token_row = ttk.Frame(outer)
        token_row.grid(row=7, column=0, columnspan=3, sticky="ew")
        btn(token_row, "Сохранить токен", self._save_token).pack(side="left")
        self.token_status = tk.StringVar(value="✅ Токен сохранён" if token else "")
        ttk.Label(token_row, textvariable=self.token_status).pack(side="left", padx=12)

        ttk.Separator(outer).grid(row=8, column=0, columnspan=3, sticky="ew", pady=12)

        # ── Лицензии ──────────────────────────────────────────────────────
        ttk.Label(outer, text="📋  Лицензии моделей", font=("System", 13, "bold")).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(0, 6))

        self.license_vars = []
        for i, (name, model) in enumerate(HF_LICENSES):
            var = tk.StringVar(value=f"○  {name}")
            self.license_vars.append(var)

            ttk.Label(outer, textvariable=var, width=42, anchor="w").grid(
                row=10 + i, column=0, columnspan=2, sticky="w", pady=2)

            url = f"https://huggingface.co/{model}"
            link_label(outer, "Открыть →", url=url).grid(
                row=10 + i, column=2, sticky="e", pady=2)

        lic_row = ttk.Frame(outer)
        lic_row.grid(row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.check_btn = btn(lic_row, "Проверить статус", self._check_licenses)
        self.check_btn.pack(side="left")

        if not token:
            for var in self.license_vars:
                var.set("🔒  Введите токен для проверки")
            self.check_btn.state(["disabled"])

        ttk.Separator(outer).grid(row=14, column=0, columnspan=3, sticky="ew", pady=12)

        btn(outer, "Закрыть", self.root.destroy).grid(row=15, column=2, sticky="e")

    # ── Папки ─────────────────────────────────────────────────────────────

    def _change_meeting(self):
        path = filedialog.askdirectory(title="Выберите папку для транскриптов встреч")
        if path:
            cfg = load_config()
            cfg["output_dir"] = path
            save_config(cfg)
            self.meeting_var.set(path)

    def _change_note(self):
        path = filedialog.askdirectory(title="Выберите папку для аудиозаметок")
        if path:
            cfg = load_config()
            cfg["note_dir"] = path
            save_config(cfg)
            self.note_var.set(path)

    # ── Токен ─────────────────────────────────────────────────────────────

    def _save_token(self):
        token = self.token_var.get().strip()
        if not token.startswith("hf_") or len(token) < 20:
            self.token_status.set("❌ Токен должен начинаться с hf_")
            return
        save_token(token)
        self.token_status.set("✅ Токен сохранён")
        self._unlock_licenses()

    def _unlock_licenses(self):
        for i, (name, _) in enumerate(HF_LICENSES):
            if "🔒" in self.license_vars[i].get():
                self.license_vars[i].set(f"○  {name}")
        self.check_btn.state(["!disabled"])

    # ── Лицензии ──────────────────────────────────────────────────────────

    def _check_licenses(self):
        token = load_token()
        if not token:
            return
        for i, (name, _) in enumerate(HF_LICENSES):
            self.license_vars[i].set(f"⏳  {name}")
        self.check_btn.state(["disabled"])
        threading.Thread(target=self._check_licenses_async, args=(token,), daemon=True).start()

    def _check_licenses_async(self, token):
        for i, (name, model) in enumerate(HF_LICENSES):
            try:
                r = requests.get(
                    f"https://huggingface.co/api/models/{model}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=8,
                )
                if r.status_code == 200:
                    text = f"✅  {name}"
                elif r.status_code == 403:
                    text = f"❌  {name} — нужно принять лицензию"
                else:
                    text = f"⚠️  {name} — ошибка {r.status_code}"
            except Exception:
                text = f"⚠️  {name} — нет соединения"
            self.root.after(0, lambda t=text, j=i: self.license_vars[j].set(t))

        self.root.after(0, lambda: self.check_btn.state(["!disabled"]))


if __name__ == "__main__":
    root = tk.Tk()
    SettingsApp(root)
    root.update()  # окно полностью инициализировано и показано

    # Прячем из дока только ПОСЛЕ того как tkinter всё настроил
    try:
        import AppKit
        AppKit.NSApplication.sharedApplication().setActivationPolicy_(
            AppKit.NSApplicationActivationPolicyAccessory)
    except Exception:
        pass

    root.mainloop()
