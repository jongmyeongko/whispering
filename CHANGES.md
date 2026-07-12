# Whispering 커스터마이징 변경 내역

이 문서는 **원본 소스**([Jemtaly/Whispering](https://github.com/Jemtaly/Whispering), `pyproject.toml` 기준 `version = 0.1.0`) 대비 이 프로젝트에서 수정한 **모든 내용**을 정리한 것입니다.

원본을 새로 내려받았을 때 아래 절차를 그대로 따라 하면 동일한 결과물을 재현할 수 있도록, 파일별로 "원본(Before) → 변경(After)"과 신규 파일의 전체 내용을 담았습니다.

---

## 0. 변경 요약

| # | 목적 | 관련 파일 |
|---|------|-----------|
| 1 | GUI 에서 `compute_type`(양자화) 선택 | `gui.py`, `whisper_impl.py` |
| 2 | 로컬 모델만 사용(모델 선택 UI 제거, 다운로드 안 함) | `gui.py` |
| 3 | Windows 에서 CUDA DLL 로드 문제 해결 | `whisper_impl.py` |
| 4 | 오디오 드랍(누락) 방지용 대용량 캡처 버퍼 | `soundcard_impl.py` |
| 5 | 처리 지연(LAG) / 오디오 드랍(AUDIO DROP) 로깅 | `whisper_impl.py`, `gui.py` |
| 6 | GUI 에 `Interval` 입력 추가 + 기본값 설정 | `gui.py` |
| 7 | 전사 결과 파일 자동 저장(확정 실시간 + Stop 시 미확정 포함) | `gui.py` |
| 8 | 저장 파일 문장 단위 줄바꿈 후처리 기능 | `gui.py` |
| 9 | 실행/배포 스크립트 (신규 파일) | `run.sh`, `run.bat`, `deploy-to-windows.sh` |
| 10 | 자동 종료 예약(10분 단위, 최대 2시간) + 남은시간 카운트다운 | `gui.py` |

> 참고: `src/whispering/core/engine.py` 는 **수정하지 않았습니다**. (`sample_time` 은 원본부터 존재하던 인자라 GUI 배선만 필요했습니다.)

---

## 1. `src/whispering/services/transcription/whisper_impl.py`

### 1-1. 상단 import 재구성 + CUDA DLL 부트스트랩 + LAG 상수

**Before (원본 상단):**

```python
from collections import deque
from typing import Literal
from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel

from whispering.core.utils import Data, Pair
from whispering.core.interfaces import (
    TranscriptionServiceFactory,
    TranscriptionService,
    LanguageCode,
)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
```

**After:** `os`, `sys`, `time` 를 먼저 import 하고, `numpy`/`faster_whisper` **import 전에** CUDA DLL 경로를 등록해야 합니다. (ctranslate2 가 추론 시점에 `PATH` 로 cuBLAS/cuDNN 을 로드하기 때문에 import 전에 `PATH` 를 세팅해야 함.)

```python
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
```

### 1-2. `compute_type` 타입/목록 추가

**Before:**

```python
WhisperModelName = Literal["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
WhisperDeviceName = Literal["auto", "cpu", "cuda"]
WHISPER_MODEL_NAMES = list(WhisperModelName.__args__)
WHISPER_DEVICE_NAMES = list(WhisperDeviceName.__args__)
```

**After:** `WhisperComputeTypeName` 과 `WHISPER_COMPUTE_TYPE_NAMES` 두 줄을 추가합니다.

```python
WhisperModelName = Literal["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large"]
WhisperDeviceName = Literal["auto", "cpu", "cuda"]
WhisperComputeTypeName = Literal["default", "auto", "int8", "int8_float16", "int8_float32", "int16", "float16", "float32"]
WHISPER_MODEL_NAMES = list(WhisperModelName.__args__)
WHISPER_DEVICE_NAMES = list(WhisperDeviceName.__args__)
WHISPER_COMPUTE_TYPE_NAMES = list(WhisperComputeTypeName.__args__)
```

### 1-3. `get_model` 에 `compute_type` 인자 추가

**Before:**

```python
@lru_cache(maxsize=None)
def get_model(model: WhisperModelName, device: WhisperDeviceName) -> WhisperModel:
    return WhisperModel(model, device)
```

**After:**

```python
@lru_cache(maxsize=None)
def get_model(
    model: WhisperModelName,
    device: WhisperDeviceName,
    compute_type: WhisperComputeTypeName,
) -> WhisperModel:
    return WhisperModel(model, device, compute_type=compute_type)
```

### 1-4. `WhisperTranscriptionService.__init__` 에 `compute_type` + LAG 상태 추가

**Before:**

```python
    def __init__(
        self,
        model: WhisperModelName,
        device: WhisperDeviceName,
        vad: bool,
        lang: LanguageCode | None,
        prompts: list[str],
        memory: int,
        patience: float,
    ):
        self.model = get_model(model, device)
        self.sample_type = np.dtype(np.float32)
        self.sample_rate = self.model.feature_extractor.sampling_rate
        self.vad = vad
        self.lang = lang
        self.prompts = deque(prompts, memory)
        self.window = np.empty((0,), dtype=self.sample_type)
        self.patience = patience
```

**After:** `compute_type` 파라미터, `get_model(...)` 인자, `self._last_lag_log = 0.0` 추가.

```python
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
```

### 1-5. `update()` 에 처리시간 측정 + LAG 로깅 추가

**Before (앞부분):**

```python
    def update(self, frame: Data) -> Pair:
        self.window = np.concatenate((self.window, frame.data))
        segments, info = self.model.transcribe(
            self.window,
            language=self.lang,
            initial_prompt="".join(self.prompts),
            vad_filter=self.vad,
        )
        segments = list(segments)
        boundary = max(len(self.window) / self.sample_rate - self.patience, 0.0)
```

**After:** 유입 오디오 길이(`added_sec`)와 처리 시간(`proc_sec`)을 재고, RTF > 1 이면 지연 로그를 출력합니다.

```python
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
```

(이후 로직은 원본과 동일합니다.)

### 1-6. `WhisperTranscriptionFactory` 에 `compute_type` 배선

**Before:** `__init__` 과 `create()` 에 `compute_type` 이 없음.

**After:** `__init__` 시그니처에 `compute_type: WhisperComputeTypeName` 추가, `self.compute_type = compute_type  # type: WhisperComputeTypeName` 저장, `create()` 의 `WhisperTranscriptionService(...)` 호출에 `compute_type=self.compute_type,` 추가.

```python
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
```

---

## 2. `src/whispering/services/audio/soundcard_impl.py`

목적: WASAPI 기본 캡처 버퍼가 매우 작아(~10-20ms), 전사 스레드가 GIL 을 오래 잡으면 버퍼가 넘쳐 오디오가 드랍됩니다(`data discontinuity`). 큰 버퍼를 요청해 일시적 지연을 흡수하도록 합니다(기본 30초, `WHISPERING_CAPTURE_BUFFER_SEC` 로 조절).

### 2-1. `os` import 추가

**Before:**

```python
from dataclasses import dataclass

import numpy as np
```

**After:**

```python
import os
from dataclasses import dataclass

import numpy as np
```

### 2-2. `SoundcardRecordingService.__init__` 에서 `blocksize` 설정

**Before:**

```python
        self.mic = mic_info.get()
        self.rec = self.mic.recorder(samplerate=sample_rate, channels=1)
        self.sample_size = int(sample_rate * sample_time)
        self.sample_type = sample_type
```

**After:**

```python
        self.mic = mic_info.get()
        # WASAPI's default capture buffer is tiny (~10-20 ms). If the reader is
        # briefly starved (e.g. while a heavy transcription holds the GIL), that
        # buffer overflows and audio is dropped ("data discontinuity"). We ask
        # for a large buffer so transient stalls are absorbed (audio queues up
        # instead of being lost). Trades potential latency for zero drops.
        # Configurable via WHISPERING_CAPTURE_BUFFER_SEC (seconds).
        try:
            buffer_sec = float(os.environ.get("WHISPERING_CAPTURE_BUFFER_SEC", "30"))
        except ValueError:
            buffer_sec = 30.0
        blocksize = max(int(sample_rate * buffer_sec), int(sample_rate * sample_time))
        self.rec = self.mic.recorder(samplerate=sample_rate, channels=1, blocksize=blocksize)
        self.sample_size = int(sample_rate * sample_time)
        self.sample_type = sample_type
```

---

## 3. `src/whispering/gui.py`

원본에서 여러 부분이 바뀌었습니다. 아래 순서대로 적용하세요.

### 3-1. import / 상수 블록

**Before (원본 상단):**

```python
import tkinter as tk
import tkinter.ttk as ttk

from whispering.core.utils import MergingQueue, Pair
from whispering.core.engine import STTEngine
from whispering.services.audio.soundcard_impl import (
    SoundcardMicrophoneInfo,
    SoundcardRecordingServiceFactory,
)
from whispering.services.transcription.whisper_impl import (
    WHISPER_MODEL_NAMES,
    WHISPER_DEVICE_NAMES,
    WHISPER_LANGUAGE_CODES,
    WhisperTranscriptionFactory,
)
from whispering.services.translation.google_impl import (
    GOOGLE_SOURCE_LANGUAGE_CODES,
    GOOGLE_TARGET_LANGUAGE_CODES,
    GoogleTranslationServiceFactory,
)

SOURCE_LANGUAGE_CODES = sorted(GOOGLE_SOURCE_LANGUAGE_CODES & WHISPER_LANGUAGE_CODES)
TARGET_LANGUAGE_CODES = sorted(GOOGLE_TARGET_LANGUAGE_CODES)
```

**After:** 표준 라이브러리 import 추가, whisper import 목록에서 `WHISPER_MODEL_NAMES` 제거 + `WHISPER_COMPUTE_TYPE_NAMES` 추가, 그리고 모델/전사 파일 경로 상수와 문장 분리 함수/전사 저장기(`TranscriptWriter`) 를 정의합니다.

```python
import os
import re
import sys
import warnings
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
from datetime import datetime
from pathlib import Path

from whispering.core.utils import MergingQueue, Pair
from whispering.core.engine import STTEngine
from whispering.services.audio.soundcard_impl import (
    SoundcardMicrophoneInfo,
    SoundcardRecordingServiceFactory,
)
from whispering.services.transcription.whisper_impl import (
    WHISPER_DEVICE_NAMES,
    WHISPER_COMPUTE_TYPE_NAMES,
    WHISPER_LANGUAGE_CODES,
    WhisperTranscriptionFactory,
)
from whispering.services.translation.google_impl import (
    GOOGLE_SOURCE_LANGUAGE_CODES,
    GOOGLE_TARGET_LANGUAGE_CODES,
    GoogleTranslationServiceFactory,
)


SOURCE_LANGUAGE_CODES = sorted(GOOGLE_SOURCE_LANGUAGE_CODES & WHISPER_LANGUAGE_CODES)
TARGET_LANGUAGE_CODES = sorted(GOOGLE_TARGET_LANGUAGE_CODES)

# Directory that holds the pre-downloaded Whisper models.
# Overridable via the WHISPERING_MODELS_DIR environment variable.
MODELS_DIR = Path(os.environ.get("WHISPERING_MODELS_DIR") or (Path.home() / "whispering-models"))

# The app uses this already-downloaded local model directory directly and never
# downloads anything. Override the full path with WHISPERING_MODEL_PATH if needed.
MODEL_PATH = Path(os.environ.get("WHISPERING_MODEL_PATH") or (MODELS_DIR / "faster-whisper-medium"))

# Directory where transcripts are saved. Each session writes confirmed text here
# incrementally, plus the final draft when Stop is pressed.
# Overridable via the WHISPERING_TRANSCRIPT_DIR environment variable.
TRANSCRIPT_DIR = Path(
    os.environ.get("WHISPERING_TRANSCRIPT_DIR") or (Path.home() / "whispering-transcripts")
)


# Sentence-ending punctuation for both Latin and CJK scripts.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？…])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Split a blob of transcript text into one sentence per element.

    Best-effort: splits on whitespace that follows sentence-ending punctuation
    (handles English, Korean, and other CJK punctuation). Blank lines in the
    source already act as hard breaks and are preserved as separators.
    """
    sentences: list[str] = []
    for block in text.splitlines():
        block = block.strip()
        if not block:
            continue
        for part in _SENTENCE_BOUNDARY.split(block):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


class TranscriptWriter:
    """Appends transcript text to a file, flushing after every write so the
    file stays up to date while transcription is running."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, text: str) -> None:
        if not text or self._file.closed:
            return
        self._file.write(text)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
```

### 3-2. `Text` 위젯: 전사 파일 저장 배선

**Before:** `__init__` 에 `writer` 없음, `update()` 에 파일 쓰기 없음.

**After:** `__init__` 에 `self.writer` 추가, `update()` 에서 확정 텍스트는 즉시 저장하고, 스트림 종료(None, Stop 시)에는 미확정 draft 까지 저장 후 파일을 닫습니다.

```python
class Text(tk.Text):
    def __init__(self, master: tk.Misc | None = None):
        super().__init__(master)
        self.result_queue = MergingQueue[Pair]()
        # Optional transcript writer. When set, confirmed text is appended as it
        # arrives, and the remaining draft is flushed + the file closed on stop.
        self.writer: TranscriptWriter | None = None
        self.tag_config("cnfm", foreground="black")
        self.tag_config("drft", foreground="blue", underline=True)
        self.insert("end", "  ", "cnfm")
        self.boundary = self.index("end-1c")
        self.see("end")
        self.config(state="disabled")
        self.update()

    def update(self):
        while self.result_queue:
            self.config(state="normal")
            if result := self.result_queue.get():
                cnfm = result.cnfm
                drft = result.drft
                self.delete(self.boundary, "end")
                self.insert("end", cnfm, "cnfm")
                self.boundary = self.index("end-1c")
                self.insert("end", drft, "drft")
                if self.writer is not None:
                    self.writer.write(cnfm)
            else:
                cnfm = self.get(self.boundary, "end-1c")
                self.delete(self.boundary, "end")
                self.insert("end", cnfm, "cnfm")
                self.insert("end", "\n", "cnfm")
                self.insert("end", "  ", "cnfm")
                self.boundary = self.index("end-1c")
                # End of stream (Stop pressed): persist the still-unconfirmed
                # draft, then finish and close the transcript file.
                if self.writer is not None:
                    self.writer.write(cnfm)
                    self.writer.write("\n")
                    self.writer.close()
                    self.writer = None
            self.see("end")
            self.config(state="disabled")
        self.after(100, self.update)  # avoid busy waiting
```

### 3-3. `App.__init__`: 모델 선택 UI 제거, Compute type / Interval 추가, 기본값 변경

**Before (원본의 해당 위젯 정의부):**

```python
        self.model_label = ttk.Label(self.head_frame, text="Model size or path:")
        self.model_combo = ttk.Combobox(self.head_frame, values=WHISPER_MODEL_NAMES, state="normal")
        self.model_combo.set("")
        self.device_label = ttk.Label(self.head_frame, text="Device:")
        self.device_combo = ttk.Combobox(self.head_frame, values=WHISPER_DEVICE_NAMES, state="readonly")
        self.device_combo.current(0)
        self.vad_check = ttk.Checkbutton(self.head_frame, text="VAD", onvalue=True, offvalue=False)
        self.vad_check.state(("!alternate", "selected"))
        self.memory_label = ttk.Label(self.head_frame, text="Memory:")
        self.memory_spin = ttk.Spinbox(self.head_frame, from_=1, to=10, increment=1, state="readonly")
        self.memory_spin.set(3)
        self.patience_label = ttk.Label(self.head_frame, text="Patience:")
        self.patience_spin = ttk.Spinbox(self.head_frame, from_=1.0, to=20.0, increment=0.5, state="readonly")
        self.patience_spin.set(5.0)
        self.timeout_label = ttk.Label(self.head_frame, text="Timeout:")
        self.timeout_spin = ttk.Spinbox(self.head_frame, from_=1.0, to=20.0, increment=0.5, state="readonly")
        self.timeout_spin.set(5.0)
```

**After:** `model_label` / `model_combo` 를 **삭제**하고, `device` 기본값을 `cuda` 로, `Compute type` 콤보박스(기본 `int8_float16`), `Interval` 스핀박스(0.1~3.0, 기본 3.0)를 추가합니다.

```python
        self.device_label = ttk.Label(self.head_frame, text="Device:")
        self.device_combo = ttk.Combobox(self.head_frame, values=WHISPER_DEVICE_NAMES, state="readonly")
        self.device_combo.set("cuda")
        self.compute_type_label = ttk.Label(self.head_frame, text="Compute type:")
        self.compute_type_combo = ttk.Combobox(self.head_frame, values=WHISPER_COMPUTE_TYPE_NAMES, state="readonly")
        self.compute_type_combo.set("int8_float16")
        self.vad_check = ttk.Checkbutton(self.head_frame, text="VAD", onvalue=True, offvalue=False)
        self.vad_check.state(("!alternate", "selected"))
        self.memory_label = ttk.Label(self.head_frame, text="Memory:")
        self.memory_spin = ttk.Spinbox(self.head_frame, from_=1, to=10, increment=1, state="readonly")
        self.memory_spin.set(3)
        self.patience_label = ttk.Label(self.head_frame, text="Patience:")
        self.patience_spin = ttk.Spinbox(self.head_frame, from_=1.0, to=20.0, increment=0.5, state="readonly")
        self.patience_spin.set(5.0)
        self.timeout_label = ttk.Label(self.head_frame, text="Timeout:")
        self.timeout_spin = ttk.Spinbox(self.head_frame, from_=1.0, to=20.0, increment=0.5, state="readonly")
        self.timeout_spin.set(5.0)
        self.interval_label = ttk.Label(self.head_frame, text="Interval:")
        self.interval_spin = ttk.Spinbox(self.head_frame, from_=0.1, to=3.0, increment=0.1, state="readonly")
        self.interval_spin.set(3.0)
```

### 3-4. `App.__init__`: head_frame `pack` 순서 변경

**Before:**

```python
        self.mic_label.pack(side="left", padx=(5, 5))
        self.mic_combo.pack(side="left", padx=(0, 5))
        self.mic_button.pack(side="left", padx=(0, 5))
        self.model_label.pack(side="left", padx=(5, 5))
        self.model_combo.pack(side="left", padx=(0, 5), fill="x", expand=True)
        self.device_label.pack(side="left", padx=(5, 5))
        self.device_combo.pack(side="left", padx=(0, 5))
        self.vad_check.pack(side="left", padx=(0, 5))
        self.memory_label.pack(side="left", padx=(5, 5))
        self.memory_spin.pack(side="left", padx=(0, 5))
        self.patience_label.pack(side="left", padx=(5, 5))
        self.patience_spin.pack(side="left", padx=(0, 5))
        self.timeout_label.pack(side="left", padx=(5, 5))
        self.timeout_spin.pack(side="left", padx=(0, 5))
```

**After:** `model_*` pack 삭제, `compute_type_*` 와 `interval_*` pack 추가.

```python
        self.mic_label.pack(side="left", padx=(5, 5))
        self.mic_combo.pack(side="left", padx=(0, 5))
        self.mic_button.pack(side="left", padx=(0, 5))
        self.device_label.pack(side="left", padx=(5, 5))
        self.device_combo.pack(side="left", padx=(0, 5))
        self.compute_type_label.pack(side="left", padx=(5, 5))
        self.compute_type_combo.pack(side="left", padx=(0, 5))
        self.vad_check.pack(side="left", padx=(0, 5))
        self.memory_label.pack(side="left", padx=(5, 5))
        self.memory_spin.pack(side="left", padx=(0, 5))
        self.patience_label.pack(side="left", padx=(5, 5))
        self.patience_spin.pack(side="left", padx=(0, 5))
        self.timeout_label.pack(side="left", padx=(5, 5))
        self.timeout_spin.pack(side="left", padx=(0, 5))
        self.interval_label.pack(side="left", padx=(5, 5))
        self.interval_spin.pack(side="left", padx=(0, 5))
```

### 3-5. `App.__init__`: foot_frame 에 "Split sentences" 버튼 추가

**Before:**

```python
        self.prompt_label = ttk.Label(self.foot_frame, text="Prompt:")
        self.prompt_entry = ttk.Entry(self.foot_frame, state="normal")
        self.control_button = ttk.Button(self.foot_frame)
        self.on_stopped()
        ...
        self.prompt_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        self.control_button.pack(side="left", padx=(5, 5))
```

**After:** 후처리 버튼 위젯 생성 + pack 추가.

```python
        self.prompt_label = ttk.Label(self.foot_frame, text="Prompt:")
        self.prompt_entry = ttk.Entry(self.foot_frame, state="normal")
        self.postprocess_button = ttk.Button(
            self.foot_frame, text="Split sentences", command=self.postprocess_file
        )
        self.control_button = ttk.Button(self.foot_frame)
        self.on_stopped()
        ...
        self.prompt_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        self.postprocess_button.pack(side="left", padx=(5, 5))
        self.control_button.pack(side="left", padx=(5, 5))
```

### 3-6. `mic_combo_refresh` 뒤에 `postprocess_file` 메서드 추가

원본 `mic_combo_refresh` 바로 아래에 다음 메서드를 추가합니다.

```python
    def postprocess_file(self):
        # Let the user pick a saved transcript and write a new file with one
        # sentence per line. The output keeps the original name but adds a
        # ".sentences" marker, so the source file is never overwritten.
        initial_dir = TRANSCRIPT_DIR if TRANSCRIPT_DIR.is_dir() else Path.home()
        selected = filedialog.askopenfilename(
            title="Select a transcript to split into sentences",
            initialdir=str(initial_dir),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not selected:
            return
        src = Path(selected)
        try:
            text = src.read_text(encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Post-process failed", f"Could not read file:\n{exc}")
            return
        sentences = split_into_sentences(text)
        if not sentences:
            messagebox.showwarning("Post-process", "No sentences were found in the file.")
            return
        dst = src.with_name(f"{src.stem}.sentences{src.suffix or '.txt'}")
        try:
            dst.write_text("\n".join(sentences) + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Post-process failed", f"Could not write file:\n{exc}")
            return
        print(f"[whispering] wrote {len(sentences)} sentences to: {dst}")
        messagebox.showinfo(
            "Post-process complete",
            f"Wrote {len(sentences)} sentences to:\n{dst}",
        )
```

### 3-7. `on_stopped`: 실패 시 전사 파일 안전 종료

**Before:**

```python
    def on_stopped(self, err: Exception | None = None):
        if err:
            print(err)
```

**After:**

```python
    def on_stopped(self, err: Exception | None = None):
        if err:
            print(err)
            # Startup failed before any end-of-stream marker was queued, so the
            # Text widget will not close the transcript file itself. Do it here.
            if self.transc_text.writer is not None:
                self.transc_text.writer.close()
                self.transc_text.writer = None
```

### 3-8. `start()`: 모델 경로 확인 + 전사 파일 생성 + `compute_type`/`interval` 배선

**Before (원본 `start` 전체):**

```python
        def start():
            self.control_button.config(text="Starting...", state="disabled")
            mic_factory = SoundcardRecordingServiceFactory(
                mic_info=self.mics[self.mic_combo.current()],
            )
            transc_factory = WhisperTranscriptionFactory(
                model=self.model_combo.get(),  # type: ignore
                device=self.device_combo.get(),  # type: ignore
                vad=self.vad_check.instate(["selected"]),
                prompts=[self.prompt_entry.get()],
                memory=int(self.memory_spin.get()),
                patience=float(self.patience_spin.get()),
            )
            transl_factory = GoogleTranslationServiceFactory(
                timeout=float(self.timeout_spin.get()),
            )
            STTEngine.start(
                record_factory=mic_factory,
                sample_time=0.1,
                transc_factory=transc_factory,
                transl_factory=transl_factory,
                source_lang=None if self.source_combo.get() == "auto" else self.source_combo.get(),  # type: ignore
                target_lang=None if self.target_combo.get() == "none" else self.target_combo.get(),  # type: ignore
                transc_result_queue=self.transc_text.result_queue,
                transl_result_queue=self.transl_text.result_queue,
                on_failure=self.on_stopped,
                on_success=self.on_started,
                on_stopped=self.on_stopped,
                on_record_error=print,
                on_transc_error=print,
                on_transl_error=print,
            )
```

**After:** 주요 변경점 — ① 로컬 `MODEL_PATH` 존재 검사, ② 전사 파일(`TranscriptWriter`) 생성, ③ `model=str(MODEL_PATH)`, ④ `compute_type=self.compute_type_combo.get()`, ⑤ `sample_time=float(self.interval_spin.get())`.

```python
        def start():
            if not MODEL_PATH.is_dir():
                self.on_stopped(FileNotFoundError(f"Model directory not found: {MODEL_PATH}"))
                return
            self.control_button.config(text="Starting...", state="disabled")
            transcript_path = TRANSCRIPT_DIR / (
                "transcript_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
            )
            try:
                self.transc_text.writer = TranscriptWriter(transcript_path)
                print(f"[whispering] saving transcript to: {transcript_path}")
            except OSError as exc:
                self.transc_text.writer = None
                print(f"[whispering] could not open transcript file: {exc}")
            mic_factory = SoundcardRecordingServiceFactory(
                mic_info=self.mics[self.mic_combo.current()],
            )
            transc_factory = WhisperTranscriptionFactory(
                model=str(MODEL_PATH),  # type: ignore
                device=self.device_combo.get(),  # type: ignore
                compute_type=self.compute_type_combo.get(),  # type: ignore
                vad=self.vad_check.instate(["selected"]),
                prompts=[self.prompt_entry.get()],
                memory=int(self.memory_spin.get()),
                patience=float(self.patience_spin.get()),
            )
            transl_factory = GoogleTranslationServiceFactory(
                timeout=float(self.timeout_spin.get()),
            )
            STTEngine.start(
                record_factory=mic_factory,
                sample_time=float(self.interval_spin.get()),
                transc_factory=transc_factory,
                transl_factory=transl_factory,
                source_lang=None if self.source_combo.get() == "auto" else self.source_combo.get(),  # type: ignore
                target_lang=None if self.target_combo.get() == "none" else self.target_combo.get(),  # type: ignore
                transc_result_queue=self.transc_text.result_queue,
                transl_result_queue=self.transl_text.result_queue,
                on_failure=self.on_stopped,
                on_success=self.on_started,
                on_stopped=self.on_stopped,
                on_record_error=print,
                on_transc_error=print,
                on_transl_error=print,
            )
```

### 3-9. `main()`: 오디오 드랍 로깅 훅 설치

**Before:**

```python
def main() -> None:
    App().mainloop()
```

**After:** 파일 상단(`App` 클래스 아래, `main` 위)에 `_install_capture_drop_logging` 함수를 추가하고 `main()` 에서 호출합니다.

```python
def _install_capture_drop_logging() -> None:
    # SoundCard emits a "data discontinuity in recording" warning whenever the
    # OS capture buffer overflows, i.e. some input audio was actually dropped
    # (usually because transcription can't keep up in real time). By default
    # Python shows each warning only once, so real drops go unnoticed. Here we
    # count every occurrence and log it explicitly so missing audio is visible.
    counter = {"n": 0}
    previous = warnings.showwarning

    try:
        from soundcard.mediafoundation import SoundcardRuntimeWarning  # type: ignore
        warnings.simplefilter("always", SoundcardRuntimeWarning)
    except Exception:
        pass

    def hook(message, category, filename, lineno, file=None, line=None):
        text = str(message).lower()
        name = getattr(category, "__name__", "")
        if "discontinuity" in text or "SoundcardRuntimeWarning" in name:
            counter["n"] += 1
            sys.stderr.write(
                f"[whispering] AUDIO DROP #{counter['n']}: capture discontinuity "
                f"- some input audio was lost (transcription not keeping up)\n"
            )
            sys.stderr.flush()
            return
        return previous(message, category, filename, lineno, file, line)

    warnings.showwarning = hook


def main() -> None:
    _install_capture_drop_logging()
    App().mainloop()
```

### 3-10. 자동 종료 예약 (Auto-stop)

Start 후 10분 단위(최대 120분)로 자동 Stop 을 예약하고, Stop 버튼에 남은 시간을 카운트다운으로 표시합니다.

**(a) `import time` 추가** — `gui.py` 상단 import 에 `import time` 을 포함합니다 (3-1 의 import 목록 기준 `sys` 다음).

```python
import os
import re
import sys
import time
import warnings
import tkinter as tk
```

**(b) head_frame 에 Auto-stop 위젯 추가** — `interval_spin` 정의/`pack` 바로 뒤에 추가합니다.

```python
        # 위젯 정의 (interval_spin 뒤)
        # Auto-stop timer: 0 = off, otherwise stop automatically after N minutes
        # (10-minute steps, up to 2 hours). Only read when Start is pressed.
        self.autostop_label = ttk.Label(self.head_frame, text="Auto-stop(min):")
        self.autostop_spin = ttk.Spinbox(self.head_frame, from_=0, to=120, increment=10, state="readonly")
        self.autostop_spin.set(0)

        # pack (interval_spin.pack 뒤)
        self.autostop_label.pack(side="left", padx=(5, 5))
        self.autostop_spin.pack(side="left", padx=(0, 5))
```

**(c) 타이머 상태 초기화** — `control_button` 생성 후 `self.on_stopped()` 호출 **전에** 추가합니다.

```python
        self.control_button = ttk.Button(self.foot_frame)
        # Auto-stop timer bookkeeping (scheduled with Tk's after()).
        self._stop_cmd = None
        self._autostop_after_id = None
        self._countdown_after_id = None
        self._stop_deadline = None
        self.on_stopped()
```

**(d) `on_started` 에서 타이머 예약 + 헬퍼 메서드 추가**

```python
    def on_started(self, eng: STTEngine):
        def stop():
            self._cancel_autostop()
            self.control_button.config(text="Stopping...", state="disabled")
            eng.stop()

        self._stop_cmd = stop
        self.control_button.config(text="Stop", command=stop, state="normal")

        # Schedule automatic stop if the Auto-stop timer is set (> 0 minutes).
        try:
            minutes = int(float(self.autostop_spin.get()))
        except (TypeError, ValueError):
            minutes = 0
        if minutes > 0:
            self._stop_deadline = time.monotonic() + minutes * 60
            self._autostop_after_id = self.after(minutes * 60 * 1000, self._auto_stop)
            print(f"[whispering] auto-stop scheduled in {minutes} minute(s)")
            self._update_countdown()

    def _auto_stop(self):
        self._autostop_after_id = None
        if self._stop_cmd is not None:
            print("[whispering] auto-stop timer elapsed - stopping")
            self._stop_cmd()

    def _cancel_autostop(self):
        if self._autostop_after_id is not None:
            self.after_cancel(self._autostop_after_id)
            self._autostop_after_id = None
        if self._countdown_after_id is not None:
            self.after_cancel(self._countdown_after_id)
            self._countdown_after_id = None
        self._stop_deadline = None

    def _update_countdown(self):
        self._countdown_after_id = None
        if self._stop_deadline is None:
            return
        remaining = max(int(round(self._stop_deadline - time.monotonic())), 0)
        # Only decorate the button while the Stop action is available.
        if self.control_button.cget("text").startswith("Stop") and \
                str(self.control_button.cget("state")) != "disabled":
            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)
            self.control_button.config(text=f"Stop ({h:d}:{m:02d}:{s:02d})")
        if remaining > 0 and self._autostop_after_id is not None:
            self._countdown_after_id = self.after(1000, self._update_countdown)
```

**(e) `on_stopped` 시작부에 타이머 정리 추가**

```python
    def on_stopped(self, err: Exception | None = None):
        self._cancel_autostop()
        self._stop_cmd = None
        if err:
            print(err)
            ...
```

---

## 4. 신규 파일

원본에는 없던 파일들입니다. 아래 내용 그대로 프로젝트 루트에 생성하세요.

### 4-1. `run.sh` (Linux/WSL2 실행 스크립트)

```bash
#!/usr/bin/env bash
#
# Whispering 한번에 실행 스크립트
#
# 하는 일:
#   1. 시스템 의존성(python3-tk) 확인 및 설치
#   2. 가상환경(.venv) 생성 (없으면)
#   3. 앱과 의존성 설치 (최초 1회)
#   4. (GPU 사용 시) CUDA 라이브러리 설치 및 경로 설정
#   5. medium 모델을 ~/whispering-models 에 다운로드 (없으면)
#   6. Whispering 실행
#
# 사용법:
#   ./run.sh              # GPU(CUDA) 모드로 실행 (기본)
#   ./run.sh --cpu        # CPU 전용으로 실행 (CUDA 라이브러리 설치 생략)
#   MODEL=large-v3 ./run.sh   # 다른 모델 다운로드/사용

set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
MODELS_DIR="${MODELS_DIR:-$HOME/whispering-models}"
export WHISPERING_MODELS_DIR="$MODELS_DIR"
MODEL="${MODEL:-medium}"
MODEL_REPO="Systran/faster-whisper-${MODEL}"
MODEL_PATH="${MODELS_DIR}/faster-whisper-${MODEL}"
# Tell the app exactly which local model directory to use (no UI selection).
export WHISPERING_MODEL_PATH="$MODEL_PATH"

USE_GPU=1
if [[ "${1:-}" == "--cpu" ]]; then
    USE_GPU=0
fi

log() { printf '\n\033[1;36m[run.sh]\033[0m %s\n' "$*"; }

# 1) 시스템 의존성: tkinter (pip 로 설치 불가, OS 패키지 필요)
if ! python3 -c "import tkinter" >/dev/null 2>&1; then
    log "tkinter 미설치 감지 -> python3-tk 설치 (sudo 필요)"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3-tk
    else
        echo "apt-get 이 없습니다. 배포판에 맞게 python3-tk(tkinter) 를 직접 설치하세요." >&2
        exit 1
    fi
fi

# 2) 가상환경 생성
if [[ ! -d "$VENV_DIR" ]]; then
    log "가상환경 생성: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 3) 앱 설치 (editable 설치로 항상 최신 소스 반영)
if ! python -c "import whispering, faster_whisper" >/dev/null 2>&1; then
    log "의존성 및 앱 설치"
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install huggingface_hub
else
    python -m pip install -e . --no-deps --quiet
fi

# 4) GPU 모드: CUDA 라이브러리 설치 및 경로 설정
if [[ "$USE_GPU" == "1" ]]; then
    if ! python -c "import nvidia.cublas, nvidia.cudnn" >/dev/null 2>&1; then
        log "CUDA 라이브러리(cublas, cudnn) 설치"
        python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    fi
    SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
    for sub in cublas cudnn cuda_nvrtc; do
        libdir="$SITE_PKGS/nvidia/$sub/lib"
        [[ -d "$libdir" ]] && export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
    done
    log "CUDA 라이브러리 경로 설정 완료 (GPU 모드)"
else
    log "CPU 전용 모드"
fi

# 5) 모델 다운로드 (없으면)
if [[ ! -f "$MODEL_PATH/model.bin" ]]; then
    log "모델 다운로드: $MODEL_REPO -> $MODEL_PATH"
    mkdir -p "$MODELS_DIR"
    hf download "$MODEL_REPO" --local-dir "$MODEL_PATH"
else
    log "모델 이미 존재: $MODEL_PATH"
fi

# 6) 실행
log "Whispering 실행"
exec whispering
```

### 4-2. `run.bat` (Windows 실행 스크립트)

> 주의: Windows 에서 사용하려면 개행을 **CRLF** 로 저장해야 합니다. (WSL 에서 만들면 `deploy-to-windows.sh` 가 자동 변환합니다.)

```bat
@echo off
REM ============================================================
REM  Whispering - Windows one-click launcher
REM
REM  What it does:
REM    1. Checks for Python 3.11+
REM    2. Creates a virtual environment (.venv)
REM    3. Installs the app and dependencies (first run only)
REM    4. (GPU mode) Installs CUDA libraries for faster-whisper
REM    5. Downloads the model to %USERPROFILE%\whispering-models
REM    6. Launches Whispering
REM
REM  Usage:
REM    run.bat            GPU (CUDA) mode - default, for GTX 1660
REM    run.bat --cpu      CPU-only mode (skips CUDA libraries)
REM    set MODEL=large-v3 && run.bat     use a different model
REM ============================================================
setlocal enableextensions
cd /d "%~dp0"

set "VENV_DIR=.venv"
if "%MODELS_DIR%"=="" set "MODELS_DIR=%USERPROFILE%\whispering-models"
set "WHISPERING_MODELS_DIR=%MODELS_DIR%"
if "%MODEL%"=="" set "MODEL=medium"
set "MODEL_REPO=Systran/faster-whisper-%MODEL%"
set "MODEL_PATH=%MODELS_DIR%\faster-whisper-%MODEL%"
REM Tell the app exactly which local model directory to use (no UI selection).
set "WHISPERING_MODEL_PATH=%MODEL_PATH%"

set "USE_GPU=1"
if /i "%~1"=="--cpu" set "USE_GPU=0"

REM --- 1) Python check ---
where python >nul 2>&1
if errorlevel 1 (
    echo [run] Python not found on PATH.
    echo [run] Install Python 3.11+ from https://www.python.org/downloads/
    echo [run] and be sure to check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

REM --- 2) Virtual environment (recreate if not a valid Windows venv) ---
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [run] Creating virtual environment...
    if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
    python -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo [run] Failed to create venv & pause & exit /b 1 )
)
call "%VENV_DIR%\Scripts\activate.bat"

REM --- 3) Install app + dependencies (first run only) ---
REM Editable install so the latest source is always used (avoids pip skipping
REM reinstall when the version number is unchanged, which can leave stale code).
python -c "import whispering, faster_whisper" >nul 2>&1
if errorlevel 1 (
    echo [run] Installing app and dependencies...
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install huggingface_hub
    if errorlevel 1 ( echo [run] Install failed & pause & exit /b 1 )
) else (
    python -m pip install -e . --no-deps --quiet
)

REM --- 4) GPU libraries (CUDA 12 / cuDNN 9) ---
if "%USE_GPU%"=="1" (
    python -c "import nvidia.cublas, nvidia.cudnn" >nul 2>&1
    if errorlevel 1 (
        echo [run] Installing CUDA libraries ^(cuBLAS, cuDNN 9^)...
        python -m pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
    )
) else (
    echo [run] CPU-only mode.
)

REM --- 5) Download model (first run only) ---
if not exist "%MODEL_PATH%\model.bin" (
    echo [run] Downloading model %MODEL_REPO% to "%MODEL_PATH%" ...
    if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
    hf download %MODEL_REPO% --local-dir "%MODEL_PATH%"
    if errorlevel 1 ( echo [run] Model download failed & pause & exit /b 1 )
) else (
    echo [run] Model already present: %MODEL_PATH%
)

REM --- 6) Launch ---
echo [run] Starting Whispering...
whispering
endlocal
```

### 4-3. `deploy-to-windows.sh` (WSL2 → Windows 배포 스크립트)

```bash
#!/usr/bin/env bash
#
# WSL2 -> Windows 배포 스크립트
#
# WSL2 에서 이 프로젝트를 윈도우 폴더(/mnt/c/... 또는 C:\...)로 복사합니다.
#   - 개발 아티팩트(.venv, __pycache__, .git 등)는 제외
#   - .bat 파일은 윈도우용 CRLF 개행으로 변환
#   - (옵션) WSL 에 받아둔 모델을 윈도우 사용자 폴더로 함께 복사
#
# 사용법:
#   ./deploy-to-windows.sh <윈도우_대상경로>
#   ./deploy-to-windows.sh 'C:\Whispering'
#   ./deploy-to-windows.sh /mnt/c/Users/jake/Whispering
#   ./deploy-to-windows.sh --with-model 'C:\Whispering'   # 모델도 함께 복사
#
# 대상경로 생략 시: <윈도우 사용자 홈>\Whispering-main 으로 배포합니다.

set -euo pipefail
cd "$(dirname "$0")"

SRC="$(pwd)"

log()  { printf '\n\033[1;36m[deploy]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[deploy] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[deploy] %s\033[0m\n' "$*" >&2; exit 1; }

# --- 인자 파싱 ---
WITH_MODEL=0
DEST_ARG=""
for arg in "$@"; do
    case "$arg" in
        --with-model) WITH_MODEL=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) DEST_ARG="$arg" ;;
    esac
done

# --- 윈도우 사용자 홈 탐지 (/mnt/c/Users/<user>) ---
# 주의: cmd.exe 는 WSL 경로에서 실행하면 실패하므로 /mnt/c 에서 실행한다.
win_home=""
if command -v cmd.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
    wp="$( (cd /mnt/c 2>/dev/null && cmd.exe /c "echo %USERPROFILE%" 2>/dev/null) | tr -d '\r\n' || true)"
    if [[ -n "$wp" ]]; then
        win_home="$(wslpath -u "$wp" 2>/dev/null || true)"
    fi
fi

# --- 대상 경로 결정 및 WSL 경로로 변환 ---
if [[ -z "$DEST_ARG" ]]; then
    [[ -n "$win_home" ]] || die "대상 경로를 지정하세요. 예: ./deploy-to-windows.sh 'C:\\Whispering'"
    DEST="$win_home/Whispering-main"
else
    if [[ "$DEST_ARG" =~ ^[A-Za-z]:[\\/] || "$DEST_ARG" == *"\\"* ]]; then
        command -v wslpath >/dev/null 2>&1 || die "wslpath 가 없어 윈도우 경로를 변환할 수 없습니다."
        DEST="$(wslpath -u "$DEST_ARG")"
    else
        DEST="$DEST_ARG"
    fi
fi

case "$DEST" in
    /mnt/*) : ;;
    *) warn "대상 '$DEST' 이(가) 윈도우 마운트(/mnt/...) 경로가 아닌 것 같습니다. 계속 진행합니다." ;;
esac

log "복사 원본:  $SRC"
log "복사 대상:  $DEST"
mkdir -p "$DEST"

# --- 제외 목록 ---
EXCLUDES=(
    ".venv" "venv" "__pycache__" ".git" ".github"
    "build" "dist" "*.egg-info" ".pytest_cache"
    "*.pyc" "*.pyo" ".mypy_cache" ".ruff_cache" ".vscode"
    "*:Zone.Identifier"
)

# --- 복사 (rsync 우선, 없으면 tar 파이프) ---
if command -v rsync >/dev/null 2>&1; then
    rsync_excludes=()
    for e in "${EXCLUDES[@]}"; do rsync_excludes+=(--exclude "$e"); done
    rsync -a --delete "${rsync_excludes[@]}" "$SRC"/ "$DEST"/
else
    warn "rsync 미설치 -> tar 로 복사합니다(대상 기존 파일은 유지)."
    tar_excludes=()
    for e in "${EXCLUDES[@]}"; do tar_excludes+=(--exclude="$e"); done
    tar -C "$SRC" "${tar_excludes[@]}" -cf - . | tar -C "$DEST" -xf -
fi

# --- .bat 파일 CRLF 변환 ---
log "윈도우 배치 파일(.bat) 개행을 CRLF 로 변환"
while IFS= read -r -d '' bat; do
    sed -i 's/\r$//; s/$/\r/' "$bat"
done < <(find "$DEST" -maxdepth 2 -name '*.bat' -print0)

# --- (옵션) 모델 복사 ---
if [[ "$WITH_MODEL" == "1" ]]; then
    src_models="${WHISPERING_MODELS_DIR:-$HOME/whispering-models}"
    if [[ -d "$src_models" ]]; then
        if [[ -n "$win_home" ]]; then
            dst_models="$win_home/whispering-models"
            log "모델 복사: $src_models -> $dst_models"
            mkdir -p "$dst_models"
            if command -v rsync >/dev/null 2>&1; then
                rsync -a "$src_models"/ "$dst_models"/
            else
                cp -r "$src_models"/. "$dst_models"/
            fi
        else
            warn "윈도우 사용자 홈을 찾지 못해 모델 복사를 건너뜁니다. (윈도우에서 run.bat 실행 시 자동 다운로드됩니다.)"
        fi
    else
        warn "복사할 모델이 없습니다: $src_models (윈도우에서 run.bat 실행 시 자동 다운로드됩니다.)"
    fi
fi

log "배포 완료!"
echo
echo "다음 단계 (윈도우에서):"
echo "  1) 탐색기에서 대상 폴더로 이동"
echo "  2) run.bat 더블클릭 (또는 명령창에서 run.bat)"
echo
if [[ -n "$win_home" ]]; then
    win_dest="$(command -v wslpath >/dev/null 2>&1 && wslpath -w "$DEST" 2>/dev/null || echo "$DEST")"
    echo "  윈도우 경로: $win_dest"
fi
```

---

## 5. 환경변수 정리

| 환경변수 | 기본값 | 설명 |
|----------|--------|------|
| `WHISPERING_MODELS_DIR` | `~/whispering-models` (Win: `%USERPROFILE%\whispering-models`) | 모델들이 위치한 상위 폴더 |
| `WHISPERING_MODEL_PATH` | `<MODELS_DIR>/faster-whisper-medium` | 실제로 사용할 로컬 모델 폴더(앱은 이 경로만 사용, 다운로드 안 함) |
| `WHISPERING_TRANSCRIPT_DIR` | `~/whispering-transcripts` | 전사 결과 파일 저장 폴더 |
| `WHISPERING_CAPTURE_BUFFER_SEC` | `30` | WASAPI 캡처 버퍼 크기(초). 클수록 드랍 방지, 지연 여유 ↑ |

---

## 6. 재현 절차 (요약)

1. 원본 `Whispering` 소스를 새로 내려받는다.
2. 위 **1~3** 절의 Before → After 를 각 파일에 그대로 반영한다.
3. **4** 절의 신규 파일(`run.sh`, `run.bat`, `deploy-to-windows.sh`)을 프로젝트 루트에 만든다. (`run.bat` 은 CRLF)
4. 모델을 `WHISPERING_MODEL_PATH` 위치에 미리 받아 둔다. (예: `faster-whisper-medium`)
5. 실행
   - Windows: `run.bat` (CUDA), `run.bat --cpu` (CPU)
   - Linux/WSL2: `./run.sh` / `./run.sh --cpu`
6. 동작 확인
   - GUI 기본값: Device=`cuda`, Compute type=`int8_float16`, Interval=`3.0`
   - 실행 중 콘솔에 `saving transcript to: ...` 경로 출력, 전사 파일이 실시간 생성됨
   - Stop 시 미확정 문장까지 저장됨
   - `Split sentences` 버튼 → 저장 파일 선택 → `<원본>.sentences.txt` 생성
   - `Auto-stop(min)` 을 10~120 으로 지정하면 해당 시간 후 자동 Stop, Stop 버튼에 남은 시간 카운트다운 표시
