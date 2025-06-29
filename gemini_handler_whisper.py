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
    Analyze the following conversation from a phone call. Summarize in 2-3 very brief, critical points (in point form, using bullet points) what the client wants or needs, and what the operator should do or say next. Each point should be a phrase, not a full sentence. Use clear bullet points, not numbered lists, slashes, or asterisks. Do not use asterisks for bullets or emphasis.

    Transcript: {transcript}
    """
    response = model.generate_content(prompt)
    return response.text
