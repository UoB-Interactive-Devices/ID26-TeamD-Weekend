import os

import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class SpeechMappingWorker(QObject):
    listening_started = pyqtSignal(str)
    transcript_ready = pyqtSignal(str)
    mapping_ready = pyqtSignal(str, str)
    failed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, gesture: str, macro_descriptions: dict[str, str]):
        super().__init__()
        self._gesture = gesture
        self._macro_descriptions = macro_descriptions

    @pyqtSlot()
    def run(self):
        self.listening_started.emit(self._gesture)

        spoken_text, listen_error = self._listen_and_transcribe()
        if not spoken_text:
            self.failed.emit(listen_error or "Could not capture a spoken command.")
            self.finished.emit()
            return

        self.transcript_ready.emit(spoken_text)
        mapped_macro, mapping_error = self._map_intent_to_macro(spoken_text)

        if mapped_macro == "none" or mapped_macro not in self._macro_descriptions:
            self.failed.emit(
                mapping_error or "No suitable macro was found for that spoken command."
            )
            self.finished.emit()
            return

        self.mapping_ready.emit(self._gesture, mapped_macro)
        self.finished.emit()

    def _listen_and_transcribe(self):
        recogniser = sr.Recognizer()
        recogniser.pause_threshold = 1.0

        try:
            with sr.Microphone() as source:
                recogniser.adjust_for_ambient_noise(source, duration=0.5)
                audio = recogniser.listen(source, timeout=7, phrase_time_limit=15)
                recognise = getattr(recogniser, "recognize_google", None)
                if recognise is None:
                    return None, "Speech recognition backend is unavailable."
                transcript = recognise(audio)
                return transcript, None
        except OSError as exc:
            return None, f"Microphone is unavailable: {exc}"
        except sr.WaitTimeoutError:
            return None, "Timed out waiting for speech."
        except sr.UnknownValueError:
            return None, "Speech was heard but could not be understood."
        except sr.RequestError as exc:
            return None, f"Speech recognition service error: {exc}"

    def _map_intent_to_macro(self, spoken_text: str):
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "none", "OPENAI_API_KEY is missing."

        macro_lines = "\n".join(
            [
                f"- {name}: {description}"
                for name, description in self._macro_descriptions.items()
            ]
        )

        system_prompt = (
            "You map spoken user intent to a strict list of macro keys.\n"
            "Pick exactly one macro key from the available list and reply with only that key.\n"
            "If there is no good match, reply with 'none'.\n\n"
            f"Available macros:\n{macro_lines}"
        )

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": spoken_text},
                ],
                max_tokens=20,
                temperature=0,
            )
            content = response.choices[0].message.content
            if content is None:
                return "none", "OpenAI returned an empty mapping response."
            return content.strip().lower(), None
        except Exception as exc:  # pylint: disable=broad-except
            return "none", f"OpenAI mapping failed: {exc}"
