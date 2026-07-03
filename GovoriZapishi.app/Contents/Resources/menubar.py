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
    """Оценка времени обработки — только по истории того же типа (note/meeting).

    Алгоритм: экспоненциальное взвешивание (последние записи важнее) + буфер 15%.
    Адаптируется по мере накопления истории. Не смешивает типы — заметки
    обрабатываются в разы быстрее встреч из-за отсутствия диаризации.
    """
    stats = load_stats()
    history = [e for e in stats["history"]
               if e.get("type") == rec_type and e["audio_secs"] > 0]
    if not history:
        return None  # нет истории для этого типа — показываем секундомер

    ratios = [e["processing_secs"] / e["audio_secs"] for e in history]
    n = len(ratios)

    if n == 1:
        avg_ratio = ratios[0]
    else:
        # Экспоненциальные веса: последняя запись в ~3x важнее первой при n=5
        decay = 0.80
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        avg_ratio = sum(r * w for r, w in zip(ratios, weights)) / total_w

    # +15% буфер: таймер не обнуляется раньше реального завершения
    return audio_secs * avg_ratio * 1.15


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


def _mix_streams(mic: np.ndarray, sck: np.ndarray,
                 sck_boost: float = 1.5) -> np.ndarray:
    """Смешивает mic и sck с фиксированным усилением SCK.
    sck_boost=1.5 компенсирует то, что системный звук обычно тише микрофона.
    Деление на 2 предотвращает клиппинг суммы двух сигналов."""
    has_sck = np.sqrt(np.mean(sck ** 2)) > 1e-5
    if has_sck:
        return np.clip((mic + sck * sck_boost) / 2, -1.0, 1.0)
    return mic


def open_settings_window():
    import sys
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings_window.py")
    subprocess.Popen([sys.executable, script])


# ── Приложение ─────────────────────────────────────────────────────────────────

class TranscribeApp(rumps.App):
    def __init__(self):
        super().__init__("⏺", quit_button=None)
        # Явно задаём иконку — Python не наследует AppIcon из .app бандла автоматически
        try:
            icon_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "GovoriZapishi.app", "Contents", "Resources", "AppIcon.icns"
            )
            if not os.path.exists(icon_path):
                icon_path = os.path.expanduser(
                    "~/govori-zapishi/GovoriZapishi.app/Contents/Resources/AppIcon.icns"
                )
            if os.path.exists(icon_path):
                img = AppKit.NSImage.alloc().initWithContentsOfFile_(icon_path)
                if img:
                    AppKit.NSApplication.sharedApplication().setApplicationIconImage_(img)
        except Exception:
            pass
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
        self.import_btn        = rumps.MenuItem("📂 Обработать медиафайл…", callback=self._import_media_file)
        self.process_btn       = rumps.MenuItem("▶ Обработать все записи", callback=self._process_all_pending)
        self.open_meetings_btn = rumps.MenuItem("📁 Открыть встречи",   callback=self._open_meetings)
        self.open_notes_btn    = rumps.MenuItem("📁 Открыть заметки",   callback=self._open_notes)
        self.settings_btn      = rumps.MenuItem("⚙️ Настройки",      callback=self._open_settings)
        self.quit_btn          = rumps.MenuItem("🚪 Выход",           callback=rumps.quit_application)

        self.restart_btn       = rumps.MenuItem("🔄 Перезапустить приложение", callback=self._restart_app)

        self.meeting_btn.set_callback(None)
        self.note_btn.set_callback(None)
        self.import_btn.set_callback(None)
        self.token_btn.hidden   = True
        self.process_btn.hidden = True
        self.restart_btn.hidden = True

        self.menu = [
            self.status_item, self.recording_item, self.processing_item,
            self.restart_btn, self.token_btn, None,
            self.meeting_btn, self.note_btn,
            self.import_btn,
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
        if not self._check_screen_recording():  return
        if not self._check_microphone_permission(): return
        self._ui(lambda: setattr(self.restart_btn, 'hidden', True))
        os.makedirs(AUDIO_DIR, exist_ok=True)
        self._ensure_meeting_dir()
        self._ensure_note_dir()
        cleanup_old_audio()
        token = self._ensure_token()
        if not token: return
        self._load_whisper()
        if not self._load_pipeline(token): return
        self._ui(lambda: setattr(self.status_item, 'title', "Готово"))
        self._ui(lambda: self.meeting_btn.set_callback(self._toggle_meeting))
        self._ui(lambda: self.note_btn.set_callback(self._toggle_note))
        self._ui(lambda: self.import_btn.set_callback(self._import_media_file))

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

    def _check_screen_recording(self):
        from sck_audio import check_permission
        self._ui(lambda: setattr(self.status_item, 'title', "Проверяю доступ к звуку..."))
        ok = check_permission()
        # Сохраняем статус для settings_window
        cfg = load_config(); cfg["perm_sck"] = ok; save_config(cfg)
        if ok:
            return True
        def show_hint():
            self.status_item.title = "⚠️ Нет доступа к звуку — см. Настройки"
            self.restart_btn.hidden = False
        self._ui(show_hint)
        return False

    def _check_microphone_permission(self):
        # Шаг 1: AVFoundation — проверяем текущий статус (не вызывает диалог)
        avf_status = None
        try:
            from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
            avf_status = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio))
        except ImportError:
            pass

        if avf_status == 3:   # Authorized
            cfg = load_config(); cfg["perm_mic"] = True; save_config(cfg)
            return True
        if avf_status == 2:   # Denied — явный отказ пользователя
            cfg = load_config(); cfg["perm_mic"] = False; save_config(cfg)
            self._ui(lambda: setattr(self.status_item, 'title',
                "⚠️ Нет доступа к микрофону — см. Настройки"))
            self._ui(lambda: setattr(self.restart_btn, 'hidden', False))
            return False

        # Шаг 2: sounddevice-тест на нативной частоте устройства — триггерит диалог macOS
        _PERMISSION_ERRORS = ('-10814', 'not authorized', 'permission', 'denied',
                              'unauthorized', 'inputdeviceunavailablewhileusinganother')
        try:
            dev_info = sd.query_devices(kind="input")
            native_sr = int(dev_info.get("default_samplerate", 48000))
            frames    = max(160, native_sr // 100)  # ~10 мс
            with sd.InputStream(channels=1, samplerate=native_sr, dtype='float32') as stream:
                data, _ = stream.read(frames)
            # Проверяем что получили реальный сигнал (не тишину macOS при запрете)
            rms = float(np.sqrt(np.mean(data ** 2)))
            print(f"[mic_permission] test OK sr={native_sr} rms={rms:.5f}", flush=True)
            cfg = load_config(); cfg["perm_mic"] = True; save_config(cfg)
            return True
        except Exception as e:
            err = str(e).lower()
            print(f"[mic_permission] test failed: {e}", flush=True)
            is_denied = any(k in err for k in _PERMISSION_ERRORS)
            if is_denied:
                cfg = load_config(); cfg["perm_mic"] = False; save_config(cfg)
                self._ui(lambda: setattr(self.status_item, 'title',
                    "⚠️ Нет доступа к микрофону — см. Настройки"))
                self._ui(lambda: setattr(self.restart_btn, 'hidden', False))
                return False
            print("[mic_permission] non-permission error, allowing startup", flush=True)
            cfg = load_config(); cfg["perm_mic"] = True; save_config(cfg)
            return True

    def _restart_app(self, _):
        subprocess.Popen(["open", "-a", "GovoriZapishi"])
        rumps.quit_application()

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
        MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo")
        label = "Загружаю Whisper MLX..." if os.path.exists(hf_cache) else "Скачиваю Whisper large-v3-turbo (~1.6 ГБ)..."
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

        # Определяем объём RAM: на ≤8 ГБ включаем экономию памяти
        try:
            total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        except Exception:
            total_ram = 16 * 1024 ** 3  # безопасный дефолт
        self._low_ram = total_ram <= 8 * 1024 ** 3

        if self._low_ram:
            try:
                import mlx.core
                mlx.core.metal.set_cache_limit(1 * 1024 ** 3)
                print(f"[memory] {total_ram // 1024**3}GB RAM — MLX cache capped at 1GB", flush=True)
            except Exception:
                pass

        try:
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
            self.pipeline.to(torch.device("mps"))

            # Батчинг сегментации и эмбеддингов — дефолт pyannote = 1 (чанки по одному
            # через GPU, ужасная утилизация MPS). Батч ускоряет без потери точности.
            # На ≤8 ГБ берём меньше чтобы не переполнить unified memory.
            batch = 8 if self._low_ram else 32
            try:
                self.pipeline.segmentation_batch_size = batch
                self.pipeline.embedding_batch_size = batch
                print(f"[perf] pyannote batch_size set to {batch}", flush=True)
            except Exception as be:
                print(f"[perf] could not set batch_size: {be}", flush=True)

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

    def _resolve_mic_device(self):
        """Возвращает device_id микрофона из конфига или None (системный дефолт)."""
        mic_name = load_config().get("mic_device")
        if not mic_name:
            return None
        devices = [(i, d) for i, d in enumerate(sd.query_devices())
                   if d["max_input_channels"] > 0]
        # 1. Точное совпадение
        for i, d in devices:
            if d["name"] == mic_name:
                return i
        # 2. Fuzzy: сохранённое имя содержится в названии устройства или наоборот
        mic_lower = mic_name.lower()
        for i, d in devices:
            dev_lower = d["name"].lower()
            if mic_lower in dev_lower or dev_lower in mic_lower:
                print(f"[mic] fuzzy match: {mic_name!r} → {d['name']!r}", flush=True)
                return i
        # 3. Не найдено — используем системный дефолт
        print(f"[mic] device {mic_name!r} not found, using default", flush=True)
        return None

    def _record_loop(self):
        try:
            self._record_loop_inner()
        except Exception as e:
            err = str(e)
            print(f"[record_loop] error: {err}", flush=True)
            def show_err():
                self.recording = False
                self.recording_item.hidden = True
                self.status_item.hidden = False
                self.meeting_btn.title = "🤝 Записать встречу"
                self.note_btn.title    = "📝 Записать заметку"
                self.meeting_btn.set_callback(self._toggle_meeting)
                self.note_btn.set_callback(self._toggle_note)
                self._update_app_title()
                rumps.alert("Ошибка записи", err)
            self._ui(show_err)

    def _record_loop_inner(self):
        from collections import deque
        from sck_audio import SCKCapture

        rec_type  = self.recording_type
        device_id = self._resolve_mic_device()
        cfg_mic   = load_config().get("mic_device", "(дефолт)")

        # Количество каналов и нативная частота устройства
        if device_id is not None:
            dev_info = sd.query_devices(device_id)
        else:
            dev_info = sd.query_devices(kind="input")
        n_ch    = max(1, int(dev_info["max_input_channels"]))
        dev_sr  = int(dev_info["default_samplerate"])
        print(f"[record] cfg_mic={cfg_mic!r} → device={device_id} "
              f"name={dev_info['name']!r} ch={n_ch} sr={dev_sr}", flush=True)

        # ── SCK: системный звук только для встреч ──────────────────────────
        sck_buf  = deque()
        sck_lock = threading.Lock()
        sck      = None

        if rec_type == "meeting":
            def _on_sck_chunk(chunk):
                # chunk: float32 (N, 2) → mono
                mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk.flatten()
                with sck_lock:
                    sck_buf.extend(mono.tolist())

            sck = SCKCapture()
            try:
                sck.start(_on_sck_chunk)
            except RuntimeError as e:
                print(f"[sck] warning: {e}", flush=True)
                sck = None

        # ── Микрофон + микширование ────────────────────────────────────────
        # Открываем на нативной частоте устройства (AirPods HFP = 24kHz и т.д.)
        # и ресемплируем до SAMPLE_RATE перед смешиванием
        with sd.InputStream(device=device_id, samplerate=dev_sr,
                            channels=n_ch, dtype="float32") as stream:
            while self.recording:
                mic_chunk, _ = stream.read(dev_sr)  # 1 сек при нативной частоте
                mic_mono = (mic_chunk.mean(axis=1)
                            if mic_chunk.ndim > 1 else mic_chunk.flatten())
                # Ресемпл если нативная частота отличается от 48 kHz
                if dev_sr != SAMPLE_RATE:
                    target_len = SAMPLE_RATE  # ровно 1 сек при 48 kHz
                    mic_mono = np.interp(
                        np.linspace(0, len(mic_mono) - 1, target_len),
                        np.arange(len(mic_mono)),
                        mic_mono
                    ).astype(np.float32)

                if sck is not None:
                    n = len(mic_mono)
                    with sck_lock:
                        avail = len(sck_buf)
                        if avail >= n:
                            sck_arr = np.array(
                                [sck_buf.popleft() for _ in range(n)], dtype=np.float32
                            )
                        elif avail > 0:
                            sck_arr = np.array(list(sck_buf), dtype=np.float32)
                            sck_buf.clear()
                            sck_arr = np.pad(sck_arr, (0, n - len(sck_arr)))
                        else:
                            sck_arr = np.zeros(n, dtype=np.float32)
                    mixed = _mix_streams(mic_mono, sck_arr)
                    # Диагностика каждые ~5 сек
                    if len(self.recorded) % 5 == 0:
                        mic_rms = float(np.sqrt(np.mean(mic_mono ** 2)))
                        sck_rms = float(np.sqrt(np.mean(sck_arr ** 2)))
                        print(f"[record] mic_rms={mic_rms:.5f} sck_rms={sck_rms:.5f} "
                              f"device={device_id}", flush=True)
                        if len(self.recorded) == 15 and mic_rms < 1e-5:
                            print("[record] WARNING: mic silent after 15s — "
                                  "check microphone permission in System Settings",
                                  flush=True)
                else:
                    mixed = mic_mono

                self.recorded.append(mixed)

        if sck:
            sck.stop()

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

        audio      = np.concatenate(self.recorded)   # уже 1D float32 mono
        audio_secs = len(audio) / SAMPLE_RATE
        timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        audio_path = os.path.join(AUDIO_DIR, f"{timestamp}.flac")

        try:
            sf.write(audio_path, audio, SAMPLE_RATE)
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

    # ── Импорт медиафайла ───────────────────────────────────────────────────────

    def _import_media_file(self, _):
        types = ('{"flac","wav","mp3","m4a","mp4","aac","ogg","mov",'
                 '"aif","aiff","opus","webm","mkv","3gp","caf"}')
        script = ('POSIX path of (choose file with prompt '
                  '"Выберите аудио- или видеофайл для расшифровки:" of type ' + types + ')')
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if r.returncode != 0:
            return  # отмена
        src = r.stdout.strip()
        if not src or not os.path.exists(src):
            return
        threading.Thread(target=self._import_worker, args=(src,), daemon=True).start()

    def _import_worker(self, src):
        os.makedirs(AUDIO_DIR, exist_ok=True)
        timestamp = "manual_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dst = os.path.join(AUDIO_DIR, f"{timestamp}.flac")

        self._ui(lambda: setattr(self.status_item, 'title', "Импорт файла — конвертация…"))
        # Транскодируем любой формат в 16кГц моно flac (что и нужно Whisper/pyannote)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", dst],
                capture_output=True, check=True,
            )
        except Exception as e:
            msg = (e.stderr.decode()[-400:] if hasattr(e, 'stderr') and e.stderr else str(e))
            self._ui(lambda m=msg: rumps.alert("Не удалось прочитать файл", m))
            self._ui(lambda: setattr(self.status_item, 'title', "Готово"))
            return

        try:
            data, sr = sf.read(dst)
            audio_secs = len(data) / sr
        except Exception:
            audio_secs = 0

        # Дата из исходного файла — попадёт в заголовок транскрипта
        try:
            start_dt = datetime.datetime.fromtimestamp(os.path.getmtime(src))
        except Exception:
            start_dt = datetime.datetime.now()

        item = {
            "type": "meeting", "audio_path": dst, "audio_secs": audio_secs,
            "timestamp": timestamp, "start_dt": start_dt.isoformat(), "status": "pending",
        }
        add_to_queue_file(item)
        with self._queue_lock:
            self._queue.append(item)
        self._ui(self._refresh_recordings_menu)
        self._ui(self._update_app_title)

        # Импортированный файл обрабатываем всегда (независимо от auto_process)
        if not self._processing:
            threading.Thread(target=self._process_queue, daemon=True).start()

    # ── Обработка ──────────────────────────────────────────────────────────────

    def _end_proc_activity(self):
        """Снимаем запрет сна если был установлен."""
        act = getattr(self, '_proc_activity', None)
        if act:
            try:
                from Foundation import NSProcessInfo
                NSProcessInfo.processInfo().endActivity_(act)
            except Exception:
                pass
            self._proc_activity = None

    def _process_queue(self):
        self._processing = True
        self._ui(self._refresh_recordings_menu)
        try:
            while True:
                with self._queue_lock:
                    if not self._queue: break
                    item = self._queue.pop(0)
                self._process_item(item)
        except Exception as e:
            print(f"[process_queue] unexpected error: {e}", flush=True)
        finally:
            self._end_proc_activity()  # на случай ошибки в process_item
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

        # Запрещаем сон Mac на время обработки — иначе MPS-контекст рвётся
        try:
            from Foundation import NSProcessInfo
            _NSActivityLatencyCritical = 0xFF00000000
            _NSActivityUserInitiated   = 0x00FFFFFF
            self._proc_activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                _NSActivityLatencyCritical | _NSActivityUserInitiated,
                "GovoriZapishi: transcription processing"
            )
        except Exception:
            self._proc_activity = None

        def start_timer():
            if self._proc_timer: self._proc_timer.stop()
            self._proc_timer = rumps.Timer(self._tick_processing, 1)
            self._proc_timer.start()
            self.processing_item.hidden = False
            self.status_item.hidden = True
        self._ui(start_timer)

        import time as _time
        _t_whisper0 = _time.monotonic()
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
        _whisper_secs = _time.monotonic() - _t_whisper0
        print(f"[perf] whisper {_whisper_secs:.1f}s "
              f"({_whisper_secs / max(audio_secs, 1):.2f}x) audio={audio_secs/60:.1f}m",
              flush=True)

        if rec_type == 'meeting':
            _t_diar0 = _time.monotonic()
            try:
                # max_speakers=12 сужает O(n²) кластеризацию — на встречах спикеров
                # почти всегда меньше, поиск по большему числу кластеров лишний
                diarization = self.pipeline(audio_path, max_speakers=12)
            except Exception as e:
                err = str(e)
                def show_err():
                    self._proc_timer.stop()
                    self.processing_item.hidden = True
                    self.status_item.hidden = False
                    self._update_app_title()
                    rumps.alert("Ошибка диаризации", err)
                self._ui(show_err); return

            _diar_secs = _time.monotonic() - _t_diar0
            print(f"[perf] diarization {_diar_secs:.1f}s "
                  f"({_diar_secs / max(audio_secs, 1):.2f}x) audio={audio_secs/60:.1f}m",
                  flush=True)

            # Освобождаем GPU-память после диаризации — только на ≤8 ГБ,
            # где память в дефиците. На больших машинах это лишний оверхед
            # (пересоздание MPS-пула на следующей записи в очереди).
            if getattr(self, "_low_ram", False):
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass

            # pyannote 4.x возвращает DiarizeOutput (dataclass), 3.x — Annotation напрямую
            if not hasattr(diarization, 'itertracks'):
                if hasattr(diarization, 'speaker_diarization'):
                    diarization = diarization.speaker_diarization
                elif hasattr(diarization, 'diarization'):
                    diarization = diarization.diarization
                elif isinstance(diarization, dict):
                    diarization = next(iter(diarization.values()))

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

        # Разрешаем сон обратно
        self._end_proc_activity()

        mark_done_in_queue(timestamp)
        self._session_done.append(timestamp)
        save_stats(rec_type, audio_secs, (datetime.datetime.now() - self._proc_start).total_seconds())

        basename = os.path.basename(md_path)
        subtitle  = "Встреча готова" if rec_type == 'meeting' else "Заметка готова"
        def notify(b=basename):
            self.status_item.title = f"✓ {b}"
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
