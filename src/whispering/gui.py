import os
import re
import sys
import time
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whispering")
        self.head_frame = ttk.Frame(self)
        self.body_frame = ttk.Frame(self)
        self.foot_frame = ttk.Frame(self)
        self.head_frame.pack(side="top", fill="x")
        self.body_frame.pack(side="top", fill="both", expand=True)
        self.foot_frame.pack(side="top", fill="x")
        self.transc_text = Text(self.body_frame)
        self.transl_text = Text(self.body_frame)
        self.transc_text.pack(side="left", fill="both", expand=True)
        self.transl_text.pack(side="left", fill="both", expand=True)
        self.mic_label = ttk.Label(self.head_frame, text="Mic:")
        self.mic_combo = ttk.Combobox(self.head_frame, state="readonly")
        self.mic_combo_refresh()
        self.mic_button = ttk.Button(self.head_frame, text="Refresh", command=self.mic_combo_refresh)
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
        # Auto-stop timer: 0 = off, otherwise stop automatically after N minutes
        # (1-minute steps, up to 2 hours). Only read when Start is pressed.
        self.autostop_label = ttk.Label(self.head_frame, text="Auto-stop(min):")
        self.autostop_spin = ttk.Spinbox(self.head_frame, from_=0, to=120, increment=1, state="readonly")
        self.autostop_spin.set(19)
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
        self.autostop_label.pack(side="left", padx=(5, 5))
        self.autostop_spin.pack(side="left", padx=(0, 5))
        self.source_label = ttk.Label(self.foot_frame, text="Source:")
        self.source_combo = ttk.Combobox(self.foot_frame, values=["auto"] + SOURCE_LANGUAGE_CODES, state="readonly")
        self.source_combo.current(0)
        self.target_label = ttk.Label(self.foot_frame, text="Target:")
        self.target_combo = ttk.Combobox(self.foot_frame, values=["none"] + TARGET_LANGUAGE_CODES, state="readonly")
        self.target_combo.current(0)
        self.prompt_label = ttk.Label(self.foot_frame, text="Prompt:")
        self.prompt_entry = ttk.Entry(self.foot_frame, state="normal")
        self.postprocess_button = ttk.Button(
            self.foot_frame, text="Split sentences", command=self.postprocess_file
        )
        self.countdown_label = ttk.Label(self.foot_frame, text="")
        self.control_button = ttk.Button(self.foot_frame)
        # Auto-stop timer bookkeeping (scheduled with Tk's after()).
        self._stop_cmd = None
        self._autostop_after_id = None
        self._countdown_after_id = None
        self._stop_deadline = None
        self.on_stopped()
        self.source_label.pack(side="left", padx=(5, 5))
        self.source_combo.pack(side="left", padx=(0, 5))
        self.target_label.pack(side="left", padx=(5, 5))
        self.target_combo.pack(side="left", padx=(0, 5))
        self.prompt_label.pack(side="left", padx=(5, 5))
        self.prompt_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)
        self.postprocess_button.pack(side="left", padx=(5, 5))
        self.countdown_label.pack(side="left", padx=(5, 5))
        self.control_button.pack(side="left", padx=(5, 5))

    def mic_combo_refresh(self):
        self.mics = SoundcardMicrophoneInfo.list_microphones()
        self.mic_combo.config(values=[f"[{mic.kind}] {mic.name}" for mic in self.mics])
        self.mic_combo.current(0)

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
        self.countdown_label.config(text="")

    def _update_countdown(self):
        self._countdown_after_id = None
        if self._stop_deadline is None:
            return
        remaining = max(int(round(self._stop_deadline - time.monotonic())), 0)
        h, rem = divmod(remaining, 3600)
        m, s = divmod(rem, 60)
        self.countdown_label.config(text=f"Auto-stop in {h:d}:{m:02d}:{s:02d}")
        if remaining > 0 and self._autostop_after_id is not None:
            self._countdown_after_id = self.after(1000, self._update_countdown)

    def on_stopped(self, err: Exception | None = None):
        self._cancel_autostop()
        self._stop_cmd = None
        if err:
            print(err)
            # Startup failed before any end-of-stream marker was queued, so the
            # Text widget will not close the transcript file itself. Do it here.
            if self.transc_text.writer is not None:
                self.transc_text.writer.close()
                self.transc_text.writer = None

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

        self.control_button.config(text="Start", command=start, state="normal")


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
