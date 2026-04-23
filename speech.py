"""
Speech-to-text via Gemini multimodal.

Accepts an audio recording and returns a verbatim transcript.
"""

from google import genai
from google.genai import types as genai_types


TRANSCRIPTION_PROMPT = """Transcribe the attached audio recording verbatim.

Rules:
- Return ONLY the transcribed text — no commentary, no timestamps, no speaker labels.
- Preserve the speaker's exact words, including any filler words.
- If the audio is silent, unintelligible, or contains no speech, return the single token: [UNINTELLIGIBLE]
- If the speaker mentions course codes (e.g. CPEN 402, CSCD 415), keep the exact spelling and spacing."""


def transcribe_audio(
    client: genai.Client,
    file_bytes: bytes,
    mime_type: str,
    model: str = "gemini-2.5-flash",
) -> str:
    """Send audio bytes to Gemini, return the plain transcript text."""
    audio_part = genai_types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
    response = client.models.generate_content(
        model=model,
        contents=[audio_part, TRANSCRIPTION_PROMPT],
    )
    transcript = (response.text or "").strip()
    return transcript
