import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput.keyboard import Controller as KeyboardController, Key
from actions import drag_start, drag_end

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.012
SILENCE_DURATION = 1.0
MIN_SPEECH_DURATION = 0.3


class VoiceTyper:
    def __init__(self, model_size="base.en"):
        self._keyboard = KeyboardController()
        self._model = None
        self._model_size = model_size
        self._running = False
        self._thread = None
        self._status = "loading"
        self._last_text = ""
        self._is_speaking = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def status(self):
        return self._status

    @property
    def last_text(self):
        return self._last_text

    @property
    def is_speaking(self):
        return self._is_speaking

    def _run(self):
        print("Loading Whisper model...", end="", flush=True)
        self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        print(" done.")
        self._status = "listening"

        chunk_sec = 0.1
        blocksize = int(SAMPLE_RATE * chunk_sec)
        silence_chunks_needed = int(SILENCE_DURATION / chunk_sec)
        min_speech_chunks = int(MIN_SPEECH_DURATION / chunk_sec)

        recording = []
        silence_count = 0
        speech_count = 0
        is_speaking = False

        def callback(indata, frames, time_info, status):
            recording.append(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                blocksize=blocksize, callback=callback):
                while self._running:
                    sd.sleep(int(chunk_sec * 1000))

                    if not recording:
                        continue

                    chunk = recording[-1]
                    rms = float(np.sqrt(np.mean(chunk ** 2)))

                    if rms > SILENCE_THRESHOLD:
                        speech_count += 1
                        silence_count = 0
                        if not is_speaking:
                            is_speaking = True
                        self._is_speaking = True
                        self._status = "hearing..."
                    else:
                        silence_count += 1

                    if is_speaking and silence_count >= silence_chunks_needed:
                        if speech_count >= min_speech_chunks:
                            audio = np.concatenate(recording)
                            self._transcribe_and_type(audio)
                        recording.clear()
                        speech_count = 0
                        silence_count = 0
                        is_speaking = False
                        self._is_speaking = False
                        self._status = "listening"

                    if not is_speaking and len(recording) > 20:
                        keep = recording[-5:]
                        recording.clear()
                        recording.extend(keep)

        except Exception as e:
            self._status = f"error: {e}"
            print(f"\nVoice error: {e}")

    def _transcribe_and_type(self, audio):
        self._status = "transcribing..."
        try:
            segments, _ = self._model.transcribe(audio, beam_size=3, language="en",
                                                  vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text and not self._is_garbage(text):
                self._last_text = text
                if not self._try_command(text):
                    self._keyboard.type(text)
                    print(f"  [voice] typed: {text}")
                self._status = f"typed: {text[:30]}"
            else:
                self._status = "listening"
        except Exception as e:
            print(f"  [voice] transcribe error: {e}")
            self._status = "listening"

    COMMANDS = {
        "enter": Key.enter,
        "return": Key.enter,
        "backspace": Key.backspace,
        "delete": Key.backspace,
        "space": Key.space,
        "tab": Key.tab,
        "escape": Key.esc,
        "undo": ("cmd", "z"),
        "redo": ("cmd", "shift", "z"),
        "select all": ("cmd", "a"),
        "copy": ("cmd", "c"),
        "paste": ("cmd", "v"),
        "cut": ("cmd", "x"),
        "drag": drag_start,
        "drop": drag_end,
    }

    def _try_command(self, text):
        lower = text.lower().strip(" .")
        for prefix in ("command ", "command-", "commando "):
            if lower.startswith(prefix):
                cmd_name = lower[len(prefix):]
                break
        else:
            return False
        action = self.COMMANDS.get(cmd_name)
        if action is None:
            return False
        if callable(action):
            action()
        elif isinstance(action, tuple):
            keys = [{"cmd": Key.cmd, "shift": Key.shift}.get(k, k) for k in action[:-1]]
            for k in keys:
                self._keyboard.press(k)
            self._keyboard.press(action[-1])
            self._keyboard.release(action[-1])
            for k in reversed(keys):
                self._keyboard.release(k)
        else:
            self._keyboard.press(action)
            self._keyboard.release(action)
        print(f"  [voice] command: {cmd_name}")
        return True

    @staticmethod
    def _is_garbage(text):
        garbage = {"thank you", "thanks for watching", "you", "the", ".", ".."}
        return text.lower().strip(" .") in garbage or len(text) < 2
