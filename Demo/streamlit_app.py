import streamlit as st
import base64
import os
import mimetypes
import struct
import logging
import io
from dotenv import load_dotenv

from google import genai
from google.genai import types

# --- Setup ---
load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key:
#     st.error("GEMINI_API_KEY is not set in your environment.")
#     st.stop()

# Setup logging (optional, for debugging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

VOICE_OPTIONS = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
    "Enceladus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia",
    "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgeunbi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
]

SYSTEM_PROMPT = """create a podcast script with two speakers
on topic = {topic}
as 
intro music: (Intro Music: Upbeat and friendly, fades slightly under the speakers' voices)
speaker 1:  Hello everyone! welcome to the podcast 
speaker 2: Hi! I am equally thrilled
"""

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict:
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def generate_script(topic: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    prompt = SYSTEM_PROMPT.format(topic=topic)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )
    result = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=contents,
        config=generate_content_config,
    ):
        result += chunk.text
    return result

def generate_audio(script: str, speaker1_voice: str, speaker2_voice: str, api_key: str) -> bytes:
    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=script)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker1_voice
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=speaker2_voice
                            )
                        ),
                    ),
                ]
            ),
        ),
    )
    audio_bytes = None
    audio_mime = None
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-flash-preview-tts",
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            audio_bytes = inline_data.data
            audio_mime = inline_data.mime_type
            break
    if audio_bytes is None:
        raise RuntimeError("No audio generated")
    file_extension = mimetypes.guess_extension(audio_mime)
    if file_extension is None:
        audio_bytes = convert_to_wav(audio_bytes, audio_mime)
    return audio_bytes

# --- Sidebar ---
st.sidebar.title("Settings")
# Remove value=... to avoid autofilling from environment variable
api_key_input = st.sidebar.text_input("Gemini API Key", type="password")
if not api_key_input:
    st.sidebar.warning("Please enter your Gemini API Key.")
api_key = api_key_input

st.sidebar.markdown("---")
st.sidebar.subheader("Speaker Voices")
speaker1_voice = st.sidebar.selectbox("Speaker 1 Voice", VOICE_OPTIONS, index=0, key="sidebar_speaker1")
speaker2_voice = st.sidebar.selectbox("Speaker 2 Voice", VOICE_OPTIONS, index=1, key="sidebar_speaker2")

# --- Main UI ---
st.title("🎙️ AI Podcast Generator")
st.title("Powered by Gemini Flash!")

with st.form("podcast_form"):
    topic = st.text_input("Podcast Topic", "")
    submitted = st.form_submit_button("Generate Podcast")

if submitted:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not topic.strip():
        st.error("Please enter a podcast topic.")
    else:
        with st.spinner("Generating podcast script and audio..."):
            try:
                script = generate_script(topic, api_key)
                st.subheader("Generated Podcast Script")
                st.code(script, language="markdown")
                audio_bytes = generate_audio(script, speaker1_voice, speaker2_voice, api_key)
                st.subheader("Podcast Audio")
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    label="Download Podcast Audio",
                    data=audio_bytes,
                    file_name="podcast.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")

