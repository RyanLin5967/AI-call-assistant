'''
Records the audio input from the microphone from the call and transcribes it to text for the Gemini API using Whisper.
Implement this code in the main UI file.
'''

# transcriber_whisper.py
import whisper
import sounddevice as sd
import numpy as np
import queue
import sys

class WhisperRecorder:
    def __init__(self, samplerate=16000, model_name="base"):
        self.samplerate = samplerate
        self.model_name = model_name
        self.q = queue.Queue()
        self.audio_data = []
        self.stream = None
        self.recording = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def start(self):
        self.audio_data = []
        self.recording = True
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, dtype='float32', callback=self._callback)
        self.stream.start()
        print("Whisper recording started...")
        while self.recording:
            self.audio_data.append(self.q.get())
        self.stream.stop()
        self.stream.close()
        print("Whisper recording stopped.")

    def stop(self):
        self.recording = False

    def transcribe(self):
        audio_np = np.concatenate(self.audio_data, axis=0).flatten()
        print("Transcribing with Whisper...")
        model = whisper.load_model(self.model_name)
        result = model.transcribe(audio_np, language='en', fp16=False, task='transcribe')
        print(f"Transcription: {result['text']}")
        return result['text'].strip()

# For direct testing
if __name__ == "__main__":
    rec = WhisperRecorder()
    import threading
    t = threading.Thread(target=rec.start)
    t.start()
    input("Press Enter to stop recording...\n")
    rec.stop()
    t.join()
    print("Transcript:", rec.transcribe())
