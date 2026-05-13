import AppKit
import rumps
import threading
import requests
import sounddevice as sd
import soundfile as sf
import numpy as np
# mlx_whisper, torch, pyannote импортируются лениво внутри _load_whisper/_load_pipeline
import datetime
import os
import subprocess
import json
from Foundation import NSOperationQueue

SAMPLE_RATE = 48000
TOKEN_FILE    = os.path.expanduser("~/.gz_token")
STATS_FILE    = os.path.expanduser("~/.gz_stats.json")
CONFIG_FILE   = os.path.expanduser("~/.gz_config.json")
QUEUE_FILE    = os.path.expanduser("~/.gz_queue.json")
DEFAULT_MEETING_DIR = os.path.expanduser("~/Documents/Stenograf/Встречи")
DEFAULT_NOTE_DIR    = os.path.expanduser("~/Documents/Stenograf/Заметки")
AUDIO_DIR     = os.path.expanduser("~/Library/Application Support/GovoriZapishi/audio")
AUDIO_RETENTION_DAYS = 7

BLACKHOLE_URL = "https://existential.audio/blackhole/"
HF_TOKEN_URL  = "https://huggingface.co/settings/tokens"
HF_LICENSES = [
    ("Диаризация спикеров",  "pyannote/speaker-diarization-3.1"),
    ("Сегментация аудио",    "pyannote/segmentation-3.0"),
    ("Сообщество диаризации","pyannote/speaker-diarization-community-1"),
]


# ── Конфиг ────────────────────────────────────────────────────────────────────

def load_config():
    if os.path.exists(CONFIG_FILE):
        return json.load(open(CONFIG_FILE))
    return {}

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

def get_meeting_dir():
    return load_config().get("output_dir", DEFAULT_MEETING_DIR)

def get_note_dir():
    return load_config().get("note_dir", DEFAULT_NOTE_DIR)

def pick_folder(prompt):
    result = subprocess.run(
        ["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        path = result.stdout.strip()
        os.makedirs(path, exist_ok=True)
        return path
    return None


# ── Статистика ─────────────────────────────────────────────────────────────────

def load_stats():
    if os.path.exists(STATS_FILE):
        return json.load(open(STATS_FILE))
    return {"history": []}

def save_stats(rec_type, audio_secs, processing_secs):
    stats = load_stats()
    stats["history"].append({"type": rec_type, "audio_secs": audio_secs, "processing_secs": processing_secs})
    stats["history"] = stats["history"][-20:]
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def estimate_processing_secs(rec_type, audio_secs):
    stats = load_stats()
    history = [e for e in stats["history"] if e.get("type") == rec_type and e["audio_secs"] > 0]
    if not history:
        history = [e for e in stats["history"] if e["audio_secs"] > 0]
    if not history:
        return None
    ratios = [e["processing_secs"] / e["audio_secs"] for e in history]
    return audio_secs * (sum(ratios) / len(ratios))


# ── Токен ──────────────────────────────────────────────────────────────────────

def load_token():
    if os.path.exists(TOKEN_FILE):
        return open(TOKEN_FILE).read().strip()
    return None

def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token.strip())


# ── Очередь (persistent) ───────────────────────────────────────────────────────

def load_queue_file():
    if os.path.exists(QUEUE_FILE):
        try:
            return json.load(open(QUEUE_FILE))
        except Exception:
            return []
    return []

def save_queue_file(items):
    with open(QUEUE_FILE, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

def add_to_queue_file(item):
    items = load_queue_file()
    serializable = dict(item)
    if isinstance(serializable.get("start_dt"), datetime.datetime):
        serializable["start_dt"] = serializable["start_dt"].isoformat()
    items.append(serializable)
    save_queue_file(items)

def mark_done_in_queue(timestamp):
    items = load_queue_file()
    for i in items:
        if i.get("timestamp") == timestamp:
            i["status"] = "done"
    save_queue_file(items)


# ── Уборка старого аудио ───────────────────────────────────────────────────────

def cleanup_old_audio():
    if not os.path.exists(AUDIO_DIR):
        return
    meeting_dir = get_meeting_dir()
    note_dir    = get_note_dir()
    cutoff = datetime.datetime.now() - datetime.timedelta(days=AUDIO_RETENTION_DAYS)
    pending_paths = {i["audio_path"] for i in load_queue_file() if i.get("status") == "pending"}
    for fname in os.listdir(AUDIO_DIR):
        if not fname.endswith(".flac"):
            continue
        audio_path = os.path.join(AUDIO_DIR, fname)
        if audio_path in pending_paths:
            continue  # не удалять файлы ожидающих обработки
        if datetime.datetime.fromtimestamp(os.path.getmtime(audio_path)) > cutoff:
            continue
        base = fname.replace(".flac", "")
        if (os.path.exists(os.path.join(meeting_dir, f"{base}.md")) or
                os.path.exists(os.path.join(note_dir, f"note_{base}.md"))):
            os.remove(audio_path)


def open_settings_window():
    import sys
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings_window.py")
    subprocess.Popen([sys.executable, script])


# ── Приложение ─────────────────────────────────────────────────────────────────

class TranscribeApp(rumps.App):
    def __init__(self):
        super().__init__("⏺", quit_button=None)
        self.recording = False
        self.recording_type = None
        self.recorded = []
        self._timer = None
        self._start_time = None
        self._proc_timer = None
        self._proc_start = None
        self._proc_estimate = None
        self.model = None
        self.pipeline = None

        self._queue = []
        self._queue_lock = threading.Lock()
        self._processing = False
        self._session_done = []      # timestamps завершённых за сессию
        self._dynamic_items = []     # rumps.MenuItem для списка записей

        # Сбросить незакрытые processing→pending при старте
        items = load_queue_file()
        dirty = any(i.get("status") == "processing" for i in items)
        if dirty:
            for i in items:
                if i.get("status") == "processing":
                    i["status"] = "pending"
            save_queue_file(items)

        # ── Статус ──
        self.status_item     = rumps.MenuItem("Инициализация...")
        self.recording_item  = rumps.MenuItem(""); self.recording_item.hidden = True
        self.processing_item = rumps.MenuItem(""); self.processing_item.hidden = True

        # ── Кнопки ──
        self.token_btn         = rumps.MenuItem("🔑 Ввести токен HuggingFace →", callback=self._prompt_token)
        self.meeting_btn       = rumps.MenuItem("🤝 Записать встречу",  callback=self._toggle_meeting)
        self.note_btn          = rumps.MenuItem("📝 Записать заметку",  callback=self._toggle_note)
        self.process_btn       = rumps.MenuItem("▶ Обработать все записи", callback=self._process_all_pending)
        self.open_meetings_btn = rumps.MenuItem("📁 Открыть встречи",   callback=self._open_meetings)
        self.open_notes_btn    = rumps.MenuItem("📁 Открыть заметки",   callback=self._open_notes)
        self.settings_btn      = rumps.MenuItem("⚙️ Настройки",      callback=self._open_settings)
        self.quit_btn          = rumps.MenuItem("🚪 Выход",           callback=rumps.quit_application)

        self.meeting_btn.set_callback(None)
        self.note_btn.set_callback(None)
        self.token_btn.hidden  = True
        self.process_btn.hidden = True

        self.menu = [
            self.status_item, self.recording_item, self.processing_item,
            self.token_btn, None,
            self.meeting_btn, self.note_btn,
            self.process_btn, None,
            self.open_meetings_btn, self.open_notes_btn, None,
            self.settings_btn, None,
            self.quit_btn,
        ]

        threading.Thread(target=self._preload, daemon=True).start()

    def _ui(self, fn):
        NSOperationQueue.mainQueue().addOperationWithBlock_(fn)

    def _open_settings(self, _):
        open_settings_window()

    # ── Список записей ─────────────────────────────────────────────────────────

    def _refresh_recordings_menu(self):
        """Динамически вставляет список записей перед status_item."""
        ns_menu = self.menu._menu

        # Удаляем старые динамические пункты
        for mi in self._dynamic_items:
            ns_menu.removeItem_(mi._menuitem)
        self._dynamic_items = []

        queue   = load_queue_file()
        pending = [i for i in queue if i.get("status") == "pending"]
        rows    = [(i, False) for i in pending]

        # Кнопка «Обработать все» — только в ручном режиме, при наличии pending, и не во время обработки
        auto_process = load_config().get("auto_process", True)
        self.process_btn.hidden = auto_process or not pending or self._processing

        if not rows:
            return

        insert_at = ns_menu.indexOfItem_(self.status_item._menuitem)

        for rec, _ in rows:
            try:
                dt = datetime.datetime.fromisoformat(rec["start_dt"])
            except Exception:
                dt = datetime.datetime.now()
            type_str = "встреча" if rec.get("type") == "meeting" else "заметка"
            title = f"→   {type_str} {dt.strftime('%d.%m %H:%M')}"
            if self._processing:
                mi = rumps.MenuItem(title, callback=lambda _: None)
                mi._menuitem.setEnabled_(False)
            else:
                ts = rec["timestamp"]
                mi = rumps.MenuItem(title, callback=lambda _, t=ts: self._process_single(t))
            ns_menu.insertItem_atIndex_(mi._menuitem, insert_at)
            self._dynamic_items.append(mi)
            insert_at += 1

    def _update_app_title(self):
        if self.recording or self._processing:
            return
        pending = [i for i in load_queue_file() if i.get("status") == "pending"]
        self.title = f"⏺{len(pending)}" if pending else "⏺"

    # ── Инициализация ──────────────────────────────────────────────────────────

    def _preload(self):
        if not self._check_blackhole():    return
        if not self._check_record_input(): return
        os.makedirs(AUDIO_DIR, exist_ok=True)
        self._ensure_meeting_dir()
        self._ensure_note_dir()
        cleanup_old_audio()
        token = self._ensure_token()
        if not token: return
        self._load_whisper()
        if not self._load_pipeline(token): return
        def _set_ready():
            self.status_item.title = "Готово"
            self.status_item.set_callback(None)
        self._ui(_set_ready)
        self._ui(lambda: self.meeting_btn.set_callback(self._toggle_meeting))
        self._ui(lambda: self.note_btn.set_callback(self._toggle_note))

        # Pending из предыдущих сессий
        queue   = load_queue_file()
        pending = [i for i in queue if i.get("status") == "pending"]
        if pending:
            self._ui(self._refresh_recordings_menu)
            self._ui(self._update_app_title)
            if load_config().get("auto_process", True):
                with self._queue_lock:
                    for item in pending:
                        if item["type"] == "note":
                            self._queue.insert(0, item)
                        else:
                            self._queue.append(item)
                threading.Thread(target=self._process_queue, daemon=True).start()

    def _check_blackhole(self):
        if any("BlackHole" in d["name"] for d in sd.query_devices()):
            return True
        self._ui(lambda: setattr(self.status_item, 'title', "⚠️ BlackHole не установлен"))
        def show():
            r = rumps.alert("BlackHole не установлен",
                f"Скачайте BlackHole 2ch и перезапустите приложение:\n{BLACKHOLE_URL}",
                ok="Открыть сайт", cancel="Закрыть")
            if r: subprocess.Popen(["open", BLACKHOLE_URL])
        self._ui(show); return False

    def _check_record_input(self):
        if any("RecordInput" in d["name"] for d in sd.query_devices()):
            return True
        self._ui(lambda: setattr(self.status_item, 'title', "⚠️ Нет устройства RecordInput"))
        def show():
            rumps.alert("Не найдено устройство RecordInput",
                "Создайте Aggregate Device с именем «RecordInput» в Audio MIDI Setup.\n"
                "Включите: ваш микрофон + BlackHole 2ch. Перезапустите приложение.")
        self._ui(show); return False

    def _ensure_meeting_dir(self):
        cfg = load_config()
        if "output_dir" not in cfg:
            path = pick_folder("Выберите папку для транскриптов встреч:") or DEFAULT_MEETING_DIR
            os.makedirs(path, exist_ok=True)
            cfg["output_dir"] = path; save_config(cfg)
        else:
            path = cfg["output_dir"]
            if not os.path.exists(path):
                def show():
                    rumps.alert("Папка встреч не найдена", f"{path}\n\nВыберите новую в ⚙️ Настройки.")
                self._ui(show)
                del cfg["output_dir"]; save_config(cfg)

    def _ensure_note_dir(self):
        cfg = load_config()
        if "note_dir" not in cfg:
            path = pick_folder("Выберите папку для аудиозаметок:") or DEFAULT_NOTE_DIR
            os.makedirs(path, exist_ok=True)
            cfg["note_dir"] = path; save_config(cfg)
        else:
            path = cfg["note_dir"]
            if not os.path.exists(path):
                def show():
                    rumps.alert("Папка заметок не найдена", f"{path}\n\nВыберите новую в ⚙️ Настройки.")
                self._ui(show)
                del cfg["note_dir"]; save_config(cfg)

    def _ensure_token(self):
        token = load_token()
        if token: return token
        self._ui(lambda: setattr(self.status_item, 'title', "⚠️ Требуется токен HuggingFace"))
        self._ui(lambda: setattr(self.token_btn, 'hidden', False))
        return None

    def _prompt_token(self, _):
        response = rumps.Window(
            title="Govori-Zapishi — Токен HuggingFace",
            message=f"Вставьте токен (тип Read).\nПолучить: {HF_TOKEN_URL}",
            default_text="hf_...", ok="Сохранить", cancel="Отмена",
            dimensions=(380, 24),
        ).run()
        if not response.clicked or not response.text.strip().startswith("hf_"): return
        save_token(response.text)
        self.token_btn.hidden = True
        self.status_item.title = "Токен сохранён, загружаю модели..."
        threading.Thread(target=self._preload, daemon=True).start()

    def _load_whisper(self):
        import mlx_whisper as _mlx
        import tempfile
        MLX_MODEL = "mlx-community/whisper-medium-mlx"
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx")
        label = "Загружаю Whisper MLX..." if os.path.exists(hf_cache) else "Скачиваю Whisper medium MLX (~500 МБ)..."
        self._ui(lambda: setattr(self.status_item, 'title', label))
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, silence, SAMPLE_RATE)
        _mlx.transcribe(tmp_path, path_or_hf_repo=MLX_MODEL)
        os.unlink(tmp_path)
        self.model = MLX_MODEL
        self._mlx = _mlx

    def _load_pipeline(self, token):
        import torch
        from pyannote.audio import Pipeline
        self._ui(lambda: setattr(self.status_item, 'title', "Загружаю модель диаризации..."))

        # Автоматическое управление памятью для ≤8 ГБ
        try:
            total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            if total_ram <= 8 * 1024 ** 3:
                import mlx.core
                mlx.core.metal.set_cache_limit(1 * 1024 ** 3)
                print(f"[memory] {total_ram // 1024**3}GB RAM — MLX cache capped at 1GB", flush=True)
        except Exception:
            pass

        try:
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
            self.pipeline.to(torch.device("mps"))
            return True
        except Exception as e:
            err = str(e)
            def show_error(err=err):
                if "401" in err or "403" in err or "gated" in err.lower() or "accept" in err.lower():
                    r = rumps.alert("Нужно принять лицензии",
                        "Откройте ⚙️ Настройки, проверьте статус лицензий и примите каждую.",
                        ok="Открыть настройки", cancel="Закрыть")
                    if r: open_settings_window()
                else:
                    rumps.alert("Ошибка загрузки модели", err)
                self.status_item.title = "⚠️ Ошибка инициализации"
            self._ui(show_error); return False

    # ── Запись ─────────────────────────────────────────────────────────────────

    def _toggle_meeting(self, _):
        if not self.recording: self._start_recording('meeting')
        else: threading.Thread(target=self._stop_recording, daemon=True).start()

    def _toggle_note(self, _):
        if not self.recording: self._start_recording('note')
        else: threading.Thread(target=self._stop_recording, daemon=True).start()

    def _start_recording(self, rec_type):
        self.recording = True
        self.recording_type = rec_type
        self.recorded = []
        self._start_time = datetime.datetime.now()
        stop_label  = "⏹ Стоп встречи" if rec_type == 'meeting' else "⏹ Стоп заметки"
        start_label = "🔴 00:00 — идёт встреча" if rec_type == 'meeting' else "🔴 00:00 — идёт заметка"
        def setup():
            if rec_type == 'meeting':
                self.meeting_btn.title = stop_label
                self.note_btn.set_callback(None)
            else:
                self.note_btn.title = stop_label
                self.meeting_btn.set_callback(None)
            self.status_item.hidden = True
            self.recording_item.title = start_label
            self.recording_item.hidden = False
        self._ui(setup)
        self._timer = rumps.Timer(self._tick_record, 1)
        self._timer.start()
        threading.Thread(target=self._record_loop, daemon=True).start()

    def _record_loop(self):
        devices   = sd.query_devices()
        device_id = next(i for i, d in enumerate(devices) if "RecordInput" in d["name"])
        with sd.InputStream(device=device_id, samplerate=SAMPLE_RATE, channels=3, dtype="float32") as stream:
            while self.recording:
                chunk, _ = stream.read(SAMPLE_RATE)
                self.recorded.append(chunk)

    def _tick_record(self, _):
        elapsed = int((datetime.datetime.now() - self._start_time).total_seconds())
        m, s = elapsed // 60, elapsed % 60
        self.title = f"🔴 {m:02d}:{s:02d}"
        suffix = "встреча" if self.recording_type == 'meeting' else "заметка"
        self.recording_item.title = f"🔴 {m:02d}:{s:02d} — идёт {suffix}"

    # ── Сохранение и очередь ───────────────────────────────────────────────────

    def _stop_recording(self):
        rec_type = self.recording_type
        self.recording = False
        self.recording_type = None
        self._timer.stop()

        audio      = np.concatenate(self.recorded)
        audio_secs = len(audio) / SAMPLE_RATE
        audio_mono = audio.mean(axis=1) if rec_type == 'meeting' else audio[:, 0]
        timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        audio_path = os.path.join(AUDIO_DIR, f"{timestamp}.flac")

        try:
            sf.write(audio_path, audio_mono, SAMPLE_RATE)
        except Exception as e:
            err = str(e)
            def show_err():
                self.recording_item.hidden = True
                self.status_item.hidden = False
                self.meeting_btn.title = "🤝 Записать встречу"
                self.note_btn.title    = "📝 Записать заметку"
                self.meeting_btn.set_callback(self._toggle_meeting)
                self.note_btn.set_callback(self._toggle_note)
                self._update_app_title()
                rumps.alert("Ошибка записи аудио", err)
            self._ui(show_err); return

        item = {
            "type":       rec_type,
            "audio_path": audio_path,
            "audio_secs": audio_secs,
            "timestamp":  timestamp,
            "start_dt":   self._start_time.isoformat(),
            "status":     "pending",
        }

        add_to_queue_file(item)

        with self._queue_lock:
            if rec_type == 'note':
                self._queue.insert(0, item)
            else:
                self._queue.append(item)

        def update_ui():
            self.recording_item.hidden = True
            self.meeting_btn.title = "🤝 Записать встречу"
            self.note_btn.title    = "📝 Записать заметку"
            self.meeting_btn.set_callback(self._toggle_meeting)
            self.note_btn.set_callback(self._toggle_note)
            self._refresh_recordings_menu()
            self._update_app_title()
        self._ui(update_ui)

        if load_config().get("auto_process", True) and not self._processing:
            threading.Thread(target=self._process_queue, daemon=True).start()

    # ── Обработка по запросу ───────────────────────────────────────────────────

    def _process_single(self, timestamp):
        """Обрабатывает одну конкретную запись по timestamp."""
        queue = load_queue_file()
        item  = next((i for i in queue
                      if i.get("timestamp") == timestamp and i.get("status") == "pending"), None)
        if not item:
            return

        if self._processing:
            # Уже идёт обработка — ставим этот элемент следующим
            with self._queue_lock:
                self._queue = [i for i in self._queue if i.get("timestamp") != timestamp]
                self._queue.insert(0, item)
            return

        # Запускаем только этот один элемент
        self._processing = True
        self._refresh_recordings_menu()  # сразу дизейблим на main thread
        def run():
            self._process_item(item)
            self._processing = False
            def done():
                self.processing_item.hidden = True
                self.status_item.hidden = False
                self._refresh_recordings_menu()
                self._update_app_title()
            self._ui(done)
        threading.Thread(target=run, daemon=True).start()

    def _process_all_pending(self, _):
        queue   = load_queue_file()
        pending = [i for i in queue if i.get("status") == "pending"]
        if not pending:
            return
        with self._queue_lock:
            existing = {i["timestamp"] for i in self._queue}
            for item in pending:
                if item["timestamp"] not in existing:
                    if item["type"] == "note":
                        self._queue.insert(0, item)
                    else:
                        self._queue.append(item)
        if not self._processing:
            threading.Thread(target=self._process_queue, daemon=True).start()

    # ── Обработка ──────────────────────────────────────────────────────────────

    def _process_queue(self):
        self._processing = True
        self._ui(self._refresh_recordings_menu)
        while True:
            with self._queue_lock:
                if not self._queue: break
                item = self._queue.pop(0)
            self._process_item(item)
        self._processing = False
        def finish_all():
            self.processing_item.hidden = True
            self.status_item.hidden = False
            self._refresh_recordings_menu()
            self._update_app_title()
        self._ui(finish_all)

    def _process_item(self, item):
        rec_type   = item["type"]
        audio_path = item["audio_path"]
        audio_secs = item["audio_secs"]
        timestamp  = item["timestamp"]
        start_dt   = item["start_dt"]
        if isinstance(start_dt, str):
            start_dt = datetime.datetime.fromisoformat(start_dt)

        self._proc_start    = datetime.datetime.now()
        self._proc_estimate = estimate_processing_secs(rec_type, audio_secs)

        def start_timer():
            if self._proc_timer: self._proc_timer.stop()
            self._proc_timer = rumps.Timer(self._tick_processing, 1)
            self._proc_timer.start()
            self.processing_item.hidden = False
            self.status_item.hidden = True
        self._ui(start_timer)

        try:
            result = self._mlx.transcribe(audio_path, path_or_hf_repo=self.model, language="ru")
        except Exception as e:
            err = str(e)
            def show_err():
                self._proc_timer.stop()
                self.processing_item.hidden = True
                self.status_item.hidden = False
                self._update_app_title()
                rumps.alert("Ошибка транскрипции", err)
            self._ui(show_err); return

        if rec_type == 'meeting':
            try:
                diarization = self.pipeline(audio_path)
            except Exception as e:
                err = str(e)
                def show_err():
                    self._proc_timer.stop()
                    self.processing_item.hidden = True
                    self.status_item.hidden = False
                    self._update_app_title()
                    rumps.alert("Ошибка диаризации", err)
                self._ui(show_err); return

            # Освобождаем GPU-память после диаризации
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass

            all_turns = [(t, sp) for t, _, sp in diarization.itertracks(yield_label=True)]
            def get_speaker(start, end):
                overlaps = {}
                for turn, sp in all_turns:
                    ov = min(turn.end, end) - max(turn.start, start)
                    if ov > 0: overlaps[sp] = overlaps.get(sp, 0) + ov
                if overlaps: return max(overlaps, key=overlaps.get)
                mid = (start + end) / 2
                return min(all_turns, key=lambda t: abs((t[0].start + t[0].end) / 2 - mid))[1] if all_turns else "UNKNOWN"

            lines = [f"# Встреча {start_dt.strftime('%d.%m.%Y')}, {start_dt.strftime('%H:%M')}", ""]
            for seg in result["segments"]:
                sp = get_speaker(seg["start"], seg["end"])
                ts = (start_dt + datetime.timedelta(seconds=seg["start"])).strftime("%H:%M:%S")
                te = (start_dt + datetime.timedelta(seconds=seg["end"])).strftime("%H:%M:%S")
                lines.append(f"**[{ts} — {te}] {sp}:** {seg['text'].strip()}")

            output_dir = get_meeting_dir()
            md_path    = os.path.join(output_dir, f"{timestamp}.md")
        else:
            lines = [f"# Аудиозаметка {start_dt.strftime('%d.%m.%Y')}, {start_dt.strftime('%H:%M')}", ""]
            for seg in result["segments"]:
                ts = (start_dt + datetime.timedelta(seconds=seg["start"])).strftime("%H:%M:%S")
                te = (start_dt + datetime.timedelta(seconds=seg["end"])).strftime("%H:%M:%S")
                lines.append(f"**[{ts} — {te}]** {seg['text'].strip()}")
            output_dir = get_note_dir()
            md_path    = os.path.join(output_dir, f"note_{timestamp}.md")

        self._ui(lambda: self._proc_timer.stop())

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as e:
            err = str(e)
            def show_err():
                self.processing_item.hidden = True
                self.status_item.hidden = False
                self._update_app_title()
                rumps.alert("Не удалось сохранить файл", err)
            self._ui(show_err); return

        mark_done_in_queue(timestamp)
        self._session_done.append(timestamp)
        save_stats(rec_type, audio_secs, (datetime.datetime.now() - self._proc_start).total_seconds())

        basename = os.path.basename(md_path)
        subtitle  = "Встреча готова" if rec_type == 'meeting' else "Заметка готова"
        def notify(p=md_path, b=basename):
            self.status_item.title = f"✓ {b}"
            self.status_item.set_callback(lambda _: subprocess.Popen(["open", p]))
            rumps.notification(title="Govori-Zapishi", subtitle=subtitle, message=b)
            self._refresh_recordings_menu()
        self._ui(notify)

    def _tick_processing(self, _):
        elapsed = int((datetime.datetime.now() - self._proc_start).total_seconds())
        if self._proc_estimate:
            remaining = max(0, int(self._proc_estimate) - elapsed)
            m, s = remaining // 60, remaining % 60
            label = f"⏳ ~{m:02d}:{s:02d} — обрабатываю"
            if not self.recording: self.title = f"⏳ ~{m:02d}:{s:02d}"
        else:
            m, s = elapsed // 60, elapsed % 60
            label = f"⏳ {m:02d}:{s:02d} — обрабатываю"
            if not self.recording: self.title = f"⏳ {m:02d}:{s:02d}"
        self.processing_item.title = label

    def _open_meetings(self, _): subprocess.Popen(["open", get_meeting_dir()])
    def _open_notes(self, _):    subprocess.Popen(["open", get_note_dir()])


if __name__ == "__main__":
    TranscribeApp().run()
