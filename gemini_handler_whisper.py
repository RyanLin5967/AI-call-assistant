'''
Fetches the transcript from the Whisper API and sends it to the Gemini API for analysis.
Implement this code in the main UI file.
'''

# gemini_handler_whisper.py
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # Loads variables from .env into environment
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-2.5-flash")

def analyze_conversation_whisper(transcript):
    prompt = f"""
    Analyze the following conversation from a phone call. In 1 or 2 very brief, critical points, summarize what the client wants or needs, and what the operator should do or say next. Be concise and actionable so the operator can respond quickly.

    Transcript: {transcript}
    """
    response = model.generate_content(prompt)
    return response.text
