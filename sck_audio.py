"""
Захват системного звука через ScreenCaptureKit (macOS 12.3+).
Не требует BlackHole, Multi-Output Device или Audio MIDI Setup.
"""
import ctypes
import threading
import numpy as np
import objc
from Foundation import NSObject

SAMPLE_RATE = 48_000
CHANNELS    = 2

# ── CoreMedia / CoreFoundation через ctypes ───────────────────────────────────

class _AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize",   ctypes.c_uint32),
        ("mData",           ctypes.c_void_p),
    ]

class _AudioBufferList(ctypes.Structure):
    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers",       _AudioBuffer * 8),
    ]

_cm = ctypes.CDLL(
    "/System/Library/Frameworks/CoreMedia.framework/CoreMedia", use_errno=True
)
_cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer.restype  = ctypes.c_int32
_cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer.argtypes = [
    ctypes.c_void_p,                 # CMSampleBufferRef
    ctypes.POINTER(ctypes.c_size_t), # bufferListSizeNeededOut — NULL
    ctypes.c_void_p,                 # AudioBufferList* (raw ptr)
    ctypes.c_size_t,                 # bufferListSize
    ctypes.c_void_p,                 # blockBufferStructureAllocator — NULL
    ctypes.c_void_p,                 # blockBufferBlockAllocator — NULL
    ctypes.c_uint32,                 # flags
    ctypes.POINTER(ctypes.c_void_p), # blockBufferOut — нам нужен для удержания данных
]

_cf = ctypes.CDLL(
    "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", use_errno=True
)
_cf.CFRelease.argtypes = [ctypes.c_void_p]
_cf.CFRelease.restype  = None


def _sample_buffer_to_numpy(sample_buffer):
    """CMSampleBuffer (PyObjC) → float32 numpy (N, channels) или None."""
    try:
        abl    = _AudioBufferList()
        bb_ref = ctypes.c_void_p(0)

        status = _cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            ctypes.c_void_p(int(sample_buffer)),
            None,
            ctypes.cast(ctypes.byref(abl), ctypes.c_void_p),
            ctypes.sizeof(abl),
            None, None, 0,
            ctypes.byref(bb_ref),
        )

        if status != 0:
            return None

        channels = []
        for i in range(min(abl.mNumberBuffers, 8)):
            b = abl.mBuffers[i]
            if b.mData and b.mDataByteSize:
                raw = (ctypes.c_byte * b.mDataByteSize).from_address(b.mData)
                channels.append(np.frombuffer(raw, dtype=np.float32).copy())

        if bb_ref.value:
            _cf.CFRelease(bb_ref)

        if not channels:
            return None

        return np.column_stack(channels) if len(channels) > 1 else channels[0].reshape(-1, 1)

    except Exception:
        return None


# ── SCStreamOutput delegate ───────────────────────────────────────────────────

class _AudioOutput(NSObject):
    def initWithCallback_(self, callback):
        self = objc.super(_AudioOutput, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
        try:
            import ScreenCaptureKit as _SCK
            if outputType != _SCK.SCStreamOutputTypeAudio:
                return
            chunk = _sample_buffer_to_numpy(sampleBuffer)
            if chunk is not None and self._callback:
                self._callback(chunk)
        except Exception:
            pass

    def stream_didStopWithError_(self, stream, error):
        pass


# ── Публичный API ─────────────────────────────────────────────────────────────

def check_permission() -> bool:
    """
    Проверяет разрешение Screen Recording.
    Первый вызов вызывает системный диалог запроса разрешения.
    Возвращает True если доступ предоставлен.
    """
    try:
        import ScreenCaptureKit as _SCK
        from Foundation import NSOperationQueue
    except ImportError:
        return False

    result = {}
    done   = threading.Event()

    def do_request():
        def handler(content, error):
            result["ok"] = (
                error is None
                and content is not None
                and len(content.displays()) > 0
            )
            done.set()
        _SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)

    # SCK completion handler требует главный run loop —
    # вызываем getWithCompletionHandler_ из главного потока
    NSOperationQueue.mainQueue().addOperationWithBlock_(do_request)
    done.wait(timeout=10.0)
    return result.get("ok", False)


class SCKCapture:
    """
    Захватывает системный звук через ScreenCaptureKit.
    chunk_callback получает float32 numpy (N, 2) из потока SCK.
    """

    def __init__(self):
        self._stream   = None
        self._output   = None
        self._callback = None
        self._started  = threading.Event()
        self._err      = None

    def start(self, chunk_callback):
        """Запускает захват. Блокирует до старта потока (≤10 сек)."""
        self._callback = chunk_callback
        self._started.clear()
        self._err = None
        try:
            import ScreenCaptureKit as _SCK
            from Foundation import NSOperationQueue
        except ImportError:
            raise RuntimeError("pyobjc-framework-ScreenCaptureKit не установлен")

        def do_request():
            _SCK.SCShareableContent.getShareableContentWithCompletionHandler_(self._on_content)

        NSOperationQueue.mainQueue().addOperationWithBlock_(do_request)
        self._started.wait(timeout=10.0)
        if self._err:
            raise RuntimeError(self._err)

    def stop(self):
        if self._stream:
            self._stream.stopCaptureWithCompletionHandler_(lambda e: None)
            self._stream = None
        self._output = None

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _on_content(self, content, error):
        import ScreenCaptureKit as _SCK

        if error or not content or not content.displays():
            self._err = (
                "Нет доступа к системному звуку.\n"
                "Разрешите в: Системные настройки → Конфиденциальность → Запись экрана"
            )
            self._started.set()
            return

        cfg = _SCK.SCStreamConfiguration.alloc().init()
        cfg.setCapturesAudio_(True)
        cfg.setExcludesCurrentProcessAudio_(False)
        cfg.setSampleRate_(SAMPLE_RATE)
        cfg.setChannelCount_(CHANNELS)
        # SCStream требует video config даже для audio-only — минимальный размер
        cfg.setWidth_(2)
        cfg.setHeight_(2)

        flt = _SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            content.displays()[0], []
        )

        self._output = _AudioOutput.alloc().initWithCallback_(self._callback)

        self._stream = _SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            flt, cfg, None
        )

        # addStreamOutput — пробуем оба варианта сигнатуры (PyObjC версии различаются)
        try:
            ok, add_err = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output, _SCK.SCStreamOutputTypeAudio, None
            )
            if not ok:
                self._err = f"SCStream addOutput failed: {add_err}"
                self._started.set()
                return
        except TypeError:
            try:
                self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self._output, _SCK.SCStreamOutputTypeAudio, None, None
                )
            except Exception as e:
                self._err = f"SCStream addOutput error: {e}"
                self._started.set()
                return

        self._stream.startCaptureWithCompletionHandler_(self._on_started)

    def _on_started(self, error):
        self._err = str(error) if error else None
        self._started.set()
