# To run this code you need to install the following dependencies:
# pip install google-genai flask

import base64
import io
import mimetypes
import os
import re
import struct
import logging
from google import genai
from google.genai import types
from flask import Flask, request, send_file, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """create a podcast script with two speakers
on topic = {topic}
as 
intro music: (Intro Music: Upbeat and friendly, fades slightly under the speakers' voices)
speaker 1:  Hello everyone! welcome to the podcast 
speaker 2: Hi! I am equally thrilled
"""


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    logger.info(f"File saved to: {file_name}")


def generate():
    logger.info("Running generate() function")
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Read aloud in a warm, welcoming tone
Speaker 1: Hello! We're excited to show you our native speech capabilities
Speaker 2: Where you can direct a voice, create realistic dialog, and so much more. Edit these placeholders to get started."""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Zephyr"
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck"
                            )
                        ),
                    ),
                ]
            ),
        ),
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
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
            file_name = f"ENTER_FILE_NAME_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            logger.info(f"Saving audio chunk to {file_name}{file_extension}")
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            logger.info("Text chunk received in generate()")
            print(chunk.text)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


@app.route("/generate_full", methods=["POST"])
def generate_full():
    logger.info("Received request at /generate_full")
    data = request.get_json()
    if not data or "topic" not in data:
        logger.warning("Missing 'topic' in request")
        return jsonify({"error": "Missing 'topic' in request"}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return jsonify({"error": "GEMINI_API_KEY environment variable not set"}), 500

    # Speaker voice selection
    allowed_voices = {
        "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe",
        "Enceladus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia",
        "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgeunbi",
        "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
    }
    speaker1_voice = data.get("speaker1_voice", "Zephyr")
    speaker2_voice = data.get("speaker2_voice", "Puck")
    if speaker1_voice not in allowed_voices:
        logger.warning(f"Invalid speaker1_voice: {speaker1_voice}, defaulting to Zephyr")
        speaker1_voice = "Zephyr"
    if speaker2_voice not in allowed_voices:
        logger.warning(f"Invalid speaker2_voice: {speaker2_voice}, defaulting to Puck")
        speaker2_voice = "Puck"

    # Step 1: Generate script
    topic = data["topic"]
    logger.info(f"Generating script for topic: {topic}")
    client = genai.Client(api_key=api_key)
    model_script = "gemini-2.0-flash"
    prompt = SYSTEM_PROMPT.format(topic=topic)
    contents_script = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config_script = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    try:
        script_result = ""
        for chunk in client.models.generate_content_stream(
            model=model_script,
            contents=contents_script,
            config=generate_content_config_script,
        ):
            script_result += chunk.text
        logger.info("Script generation successful in /generate_full")
    except Exception as e:
        logger.exception("Script generation failed in /generate_full")
        return jsonify({"error": f"Script generation failed: {str(e)}"}), 500

    # Step 2: Generate audio from script
    logger.info(f"Generating audio from script in /generate_full with voices: {speaker1_voice}, {speaker2_voice}")
    model_tts = "gemini-2.5-flash-preview-tts"
    contents_tts = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=script_result),
            ],
        ),
    ]
    generate_content_config_tts = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
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
    try:
        for chunk in client.models.generate_content_stream(
            model=model_tts,
            contents=contents_tts,
            config=generate_content_config_tts,
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
                logger.info("Audio chunk received in /generate_full")
                break
    except Exception as e:
        logger.exception("Audio generation failed in /generate_full")
        return jsonify({"error": f"Audio generation failed: {str(e)}"}), 500

    if audio_bytes is None:
        logger.error("No audio generated in /generate_full")
        return jsonify({"error": "No audio generated"}), 500

    file_extension = mimetypes.guess_extension(audio_mime)
    if file_extension is None:
        file_extension = ".wav"
        audio_bytes = convert_to_wav(audio_bytes, audio_mime)

    audio_io = io.BytesIO(audio_bytes)
    audio_io.seek(0)

    # Return both script and audio as a response
    logger.info("Returning script and audio file from /generate_full")
    from flask import send_file, make_response
    response = make_response(send_file(
        audio_io,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"output{file_extension}"
    ))
    response.headers["X-Podcast-Script"] = base64.b64encode(script_result.encode("utf-8")).decode("utf-8")
    return response


if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(host="0.0.0.0", port=8000)

