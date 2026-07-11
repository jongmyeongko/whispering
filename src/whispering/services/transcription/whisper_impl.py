import os
import sys
import time
from collections import deque
from typing import Literal
from functools import lru_cache

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Log the current transcription latency at most once every N seconds while the
# transcription is running slower than real time. Input audio is buffered (see
# the large capture buffer in the recording service), so this indicates growing
# latency, not dropped audio.
_LAG_LOG_INTERVAL = 2.0


def _add_cuda_dll_directories() -> None:
    # Windows-only: make the CUDA runtime DLLs shipped by the nvidia-*-cu12 pip
    # packages discoverable *before* importing faster-whisper / ctranslate2.
    #
    # ctranslate2 loads cuBLAS/cuDNN lazily (at inference time) via a plain
    # LoadLibrary call that searches PATH but does NOT consult directories added
    # via os.add_dll_directory(). Therefore we must prepend the nvidia "bin"
    # folders to os.environ["PATH"] as well. We do both for maximum robustness.
    # No-op on Linux/macOS (which rely on LD_LIBRARY_PATH instead).
    if os.name != "nt":
        return
    import site

    roots = list(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())

    bin_dirs = []
    seen = set()
    for root in roots:
        nvidia_dir = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for pkg in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, pkg, "bin")
            if os.path.isdir(bin_dir) and bin_dir not in seen:
                seen.add(bin_dir)
                bin_dirs.append(bin_dir)

    for bin_dir in bin_dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(bin_dir)
            except OSError:
                pass
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs + [os.environ.get("PATH", "")])


_add_cuda_dll_directories()

import numpy as np
from faster_whisper import WhisperModel

from whispering.core.utils import Data, Pair
from whispering.core.interfaces import (
    TranscriptionServiceFactory,
    TranscriptionService,
    LanguageCode,
)


WhisperModelName = Literal["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
WhisperDeviceName = Literal["auto", "cpu", "cuda"]
WhisperComputeTypeName = Literal["default", "auto", "int8", "int8_float16", "int8_float32", "int16", "float16", "float32"]
WHISPER_MODEL_NAMES = list(WhisperModelName.__args__)
WHISPER_DEVICE_NAMES = list(WhisperDeviceName.__args__)
WHISPER_COMPUTE_TYPE_NAMES = list(WhisperComputeTypeName.__args__)
WHISPER_LANGUAGE_CODES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
    "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
    "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
}


@lru_cache(maxsize=None)
def get_model(
    model: WhisperModelName,
    device: WhisperDeviceName,
    compute_type: WhisperComputeTypeName,
) -> WhisperModel:
    return WhisperModel(model, device, compute_type=compute_type)


class WhisperTranscriptionService(TranscriptionService):
    def __init__(
        self,
        model: WhisperModelName,
        device: WhisperDeviceName,
        compute_type: WhisperComputeTypeName,
        vad: bool,
        lang: LanguageCode | None,
        prompts: list[str],
        memory: int,
        patience: float,
    ):
        self.model = get_model(model, device, compute_type)
        self.sample_type = np.dtype(np.float32)
        self.sample_rate = self.model.feature_extractor.sampling_rate
        self.vad = vad
        self.lang = lang
        self.prompts = deque(prompts, memory)
        self.window = np.empty((0,), dtype=self.sample_type)
        self.patience = patience
        self._last_lag_log = 0.0

    def update(self, frame: Data) -> Pair:
        added_sec = len(frame.data) / self.sample_rate
        self.window = np.concatenate((self.window, frame.data))
        t0 = time.monotonic()
        segments, info = self.model.transcribe(
            self.window,
            language=self.lang,
            initial_prompt="".join(self.prompts),
            vad_filter=self.vad,
        )
        segments = list(segments)
        proc_sec = time.monotonic() - t0
        # RTF > 1 means this cycle took longer than the audio it ingested, so the
        # backlog (and end-to-end latency / risk of capture drops) is growing.
        rtf = proc_sec / added_sec if added_sec > 0 else 0.0
        now = time.monotonic()
        if rtf > 1.0 and now - self._last_lag_log >= _LAG_LOG_INTERVAL:
            self._last_lag_log = now
            window_sec = len(self.window) / self.sample_rate
            sys.stderr.write(
                f"[whispering] latency ~{window_sec:.1f}s behind live "
                f"(RTF={rtf:.2f}: {added_sec:.1f}s of audio took {proc_sec:.1f}s to transcribe; "
                f"slower than real time, but audio is buffered - not dropped)\n"
            )
            sys.stderr.flush()
        boundary = max(len(self.window) / self.sample_rate - self.patience, 0.0)
        i = 0
        for segment in segments:
            if segment.end >= boundary:
                if segment.start < boundary:
                    boundary = segment.start
                break
            i += 1
        cnfm_src = "".join(segment.text for segment in segments[:i])
        drft_src = "".join(segment.text for segment in segments[i:])
        self.prompts.extend(segment.text for segment in segments[:i])
        self.window = self.window[int(boundary * self.sample_rate) :]
        return Pair(cnfm_src, drft_src)

    @property
    def required_sample_type(self) -> np.dtype:
        return self.sample_type

    @property
    def required_sample_rate(self) -> int:
        return self.sample_rate


class WhisperTranscriptionFactory(TranscriptionServiceFactory):
    def __init__(
        self,
        model: WhisperModelName,
        device: WhisperDeviceName,
        compute_type: WhisperComputeTypeName,
        vad: bool,
        prompts: list[str],
        memory: int,
        patience: float,
    ):
        self.model = model  # type: WhisperModelName
        self.device = device  # type: WhisperDeviceName
        self.compute_type = compute_type  # type: WhisperComputeTypeName
        self.vad = vad
        self.prompts = prompts
        self.memory = memory
        self.patience = patience

    def create(
        self,
        lang: LanguageCode | None,
    ) -> TranscriptionService:
        return WhisperTranscriptionService(
            model=self.model,
            device=self.device,
            compute_type=self.compute_type,
            vad=self.vad,
            lang=lang,
            prompts=self.prompts,
            memory=self.memory,
            patience=self.patience,
        )
