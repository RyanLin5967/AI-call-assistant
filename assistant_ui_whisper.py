'''
Sample UI for the AI assistant using Whisper for transcription and Gemini for analysis.
Contains implementation of the gemini_handler_whisper.py and transcriber_whisper.py functionality.
'''


# assistant_ui_whisper.py
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QPalette, QColor
import sys
from transcriber_whisper import WhisperRecorder
import threading


class AnalysisWorker(QObject):
   finished = pyqtSignal(str, str)
   error = pyqtSignal(str)


   def __init__(self):
       super().__init__()
       self._transcript = None


   def analyze(self, transcript):
       try:
           from gemini_handler_whisper import analyze_conversation_whisper
           analysis = analyze_conversation_whisper(transcript)
           self.finished.emit(transcript, analysis)
       except Exception as e:
           self.error.emit(str(e))


class AssistantUI(QWidget):
   def __init__(self):
       super().__init__()
       from PyQt6.QtGui import QGuiApplication
       self.setWindowTitle("AI Call Assistant (Whisper + Gemini)")
       screen = QGuiApplication.primaryScreen().geometry()
       width = 420
       height = screen.height()
       self.resize(width, height)
       self.move(screen.width() - width, 0)  # Top right corner
       self.layout = QVBoxLayout()
       self.setStyleSheet("""
           QWidget {
               background: #181c24;
           }
           QLabel {
               color: #e3eafc;
               font-size: 22px;
               font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
               font-weight: 700;
               margin-bottom: 8px;
           }
           QTextEdit {
               background: #23283a;
               color: #e3eafc;
               border-radius: 16px;
               font-size: 20px;
               font-family: 'Fira Mono', 'Consolas', 'Menlo', monospace;
               padding: 20px;
               border: 1.5px solid #2a3142;
               margin-bottom: 12px;
               box-shadow: 0 2px 8px rgba(30, 40, 60, 0.13);
           }
           QPushButton {
               background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4f8cff, stop:1 #23283a);
               color: #fff;
               border: none;
               border-radius: 8px;
               font-size: 13px;
               font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
               font-weight: 600;
               padding: 4px 0;
               margin: 6px 0;
               min-height: 22px;
               min-width: 90px;
               letter-spacing: 0.5px;
               box-shadow: 0 2px 8px rgba(30, 40, 60, 0.18);
               transition: background 0.2s, box-shadow 0.2s;
           }
           QPushButton:hover {
               background: #3576e0;
               box-shadow: 0 4px 16px rgba(30, 40, 60, 0.23);
           }
           QPushButton:pressed {
               background: #23283a;
           }
           QPushButton:disabled {
               background: #2a3142;
               color: #888;
           }
       """)


       self.info_label = QLabel("<b>AI Call Assistant (Whisper + Gemini)</b>")
       self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
       self.info_label.setStyleSheet("font-size: 22px; color: #8ab4f8; margin: 12px 24px 0 0; padding-right: 24px;")
       self.result_box = QTextEdit()
       self.result_box.setReadOnly(True)
       self.result_box.setMinimumHeight(700)
       self.result_box.setMaximumHeight(16777215)
       self.result_box.setStyleSheet("margin-top: 32px;")


       self.start_button = QPushButton("Start Recording")
       self.stop_button = QPushButton("Stop Recording")
       self.stop_button.setEnabled(False)
       self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
       self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)


       self.start_button.clicked.connect(self.start_recording)
       self.stop_button.clicked.connect(self.stop_recording)


       self.layout.addWidget(self.info_label)
       self.layout.addWidget(self.start_button)
       self.layout.addWidget(self.stop_button)
       self.layout.addWidget(self.result_box)
       self.setLayout(self.layout)


       self.thread = QThread()
       self.worker = AnalysisWorker()
       self.worker.moveToThread(self.thread)
       self.worker.finished.connect(self.on_analysis_finished)
       self.worker.error.connect(self.on_analysis_error)
       self.thread.start()



       self._transcript = None
       self._recording_thread = None
       self._recording = False
       self._cursor_timer = None
       self._cursor_visible = True


   def start_recording(self):
       self.result_box.setHtml("<span style='color:#4f8cff;font-size:22px'><b>🔴 Recording...</b></span><br><span style='font-size:16px'>Speak now, then press <b>Stop Recording</b> to analyze.</span>")
       self.start_button.setEnabled(False)
       self.stop_button.setEnabled(True)
       self._transcript = None
       self._recorder = WhisperRecorder()
       self._recording_thread = threading.Thread(target=self._recorder.start)
       self._recording_thread.start()


   def stop_recording(self):
       self.result_box.setHtml("<span style='color:#4f8cff;font-size:22px'><b>⏳ Analyzing...</b></span><br><span style='font-size:16px'>Transcribing and analyzing, please wait.</span>")
       self.start_button.setEnabled(False)
       self.stop_button.setEnabled(False)
       if hasattr(self, '_recorder') and self._recorder:
           self._recorder.stop()
       if self._recording_thread is not None:
           self._recording_thread.join()
       try:
           transcript = self._recorder.transcribe()
           self._transcript = transcript
           self._update_caption_box(transcript)
           if self._transcript:
               # Run Gemini analysis in a background thread
               def run_analysis():
                   from gemini_handler_whisper import analyze_conversation_whisper
                   analysis = analyze_conversation_whisper(self._transcript)
                   # Use Qt signal to update UI from worker thread
                   self.worker.finished.emit(self._transcript, analysis)
               threading.Thread(target=run_analysis).start()
           else:
               self.on_analysis_error("No transcript available. Recording may have failed.")
       except Exception as e:
           self.on_analysis_error(str(e))


   def _update_caption_box(self, text):
       # Add blinking cursor effect to transcript
       from PyQt6.QtCore import QTimer
       if hasattr(self, '_cursor_timer') and self._cursor_timer:
           self._cursor_timer.stop()
           self._cursor_timer = None
       self._caption_text = text
       self._cursor_visible = True
       def update():
           cursor = "<span style='color:#4f8cff;font-weight:bold'>|</span>" if self._cursor_visible else ""
           self.result_box.setHtml(f"<b style='color:#4f8cff'>Transcript:</b> <span style='font-size:18px'>{self._caption_text}{cursor}</span>")
           self._cursor_visible = not self._cursor_visible
       self._cursor_timer = QTimer(self)
       self._cursor_timer.timeout.connect(update)
       self._cursor_timer.start(500)
       update()


   def on_analysis_finished(self, transcript, analysis):
       # Stop blinking cursor when analysis is done
       if hasattr(self, '_cursor_timer') and self._cursor_timer:
           self._cursor_timer.stop()
           self._cursor_timer = None
       # Format Gemini output: bullet points only, no numbers
       import re
       def format_bullets(text):
           # Replace any numbered or dash/numbered list with bullet points
           text = re.sub(r"^\s*([\-\d]+[\.)\-:]\s*)", "• ", text, flags=re.MULTILINE)
           # Ensure each bullet starts on a new line
           text = re.sub(r"\n?• ", r"<br>• ", text)
           return text.strip()
       formatted_analysis = format_bullets(analysis)
       self.result_box.setHtml(f"<b style='color:#4f8cff'>Transcript:</b> <span style='font-size:18px'>{transcript}</span><br><br><b style='color:#4f8cff'>Gemini Analysis:</b><br><span style='font-size:19px'>{formatted_analysis}</span>")
       self.start_button.setEnabled(True)
       self.stop_button.setEnabled(False)


   def on_analysis_error(self, error_msg):
       self.result_box.setHtml(f"<span style='color:#ff6f6f;font-size:20px'><b>❌ An error occurred:</b></span><br><span style='font-size:16px'>{error_msg}</span>")
       print(f"Error in analyze_convo: {error_msg}", file=sys.stderr)
       self.start_button.setEnabled(True)
       self.stop_button.setEnabled(False)


if __name__ == "__main__":
   app = QApplication(sys.argv)
   window = AssistantUI()
   window.show()
   sys.exit(app.exec())



