# To run this code you need to install the following dependencies:
# pip install google-genai flask

import base64
import os
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

SYSTEM_PROMPT = """create a podcast script with two speakers
on topic = {topic}
as 
intro music: (Intro Music: Upbeat and friendly, fades slightly under the speakers' voices)
speaker 1:  Hello everyone! welcome to the podcast 
speaker 2: Hi! I am equally thrilled
"""


@app.route("/generate", methods=["POST"])
def generate_podcast():
    data = request.get_json()
    if not data or "topic" not in data:
        return jsonify({"error": "Missing 'topic' in request"}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable not set"}), 500

    topic = data["topic"]
    client = genai.Client(
        api_key=api_key,
    )
    model = "gemini-2.0-flash"
    prompt = SYSTEM_PROMPT.format(topic=topic)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    try:
        result = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            result += chunk.text
        return jsonify({"script": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
