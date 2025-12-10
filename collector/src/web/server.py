import os
import json
import base64
import glob
from flask import Flask, render_template, request, jsonify, send_from_directory
from google import genai
from google.genai import types
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAoDegs3eiBWtDki2jogPfJ5sZ5fY7HMP8")
client = genai.Client(api_key=GEMINI_API_KEY)

# Output folder for Kaggle export
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dataset')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manual')
def manual_helper():
    return render_template('manual_helper.html')

@app.route('/api/log_csv', methods=['POST'])
def log_csv_entry():
    data = request.json
    csv_path = os.path.join(OUTPUT_FOLDER, 'metadata.csv')
    
    # Check if header needed
    write_header = not os.path.exists(csv_path)
    
    try:
        with open(csv_path, 'a', encoding='utf-8') as f:
            if write_header:
                f.write('id|filename|text|emotion|voice|timestamp\n')
            
            # Simple pipe delimited to avoid comma issues with Sinhala text? 
            # Or standard CSV with quoting. Let's do standard CSV manually for simplicity or use csv module.
            # Using pipe for safety with text content.
            timestamp = datetime.now().isoformat()
            line = f"{data.get('filename')}|{data.get('filename')}|{data.get('text')}|{data.get('emotion')}|{data.get('voice')}|{timestamp}\n"
            f.write(line)
            
        return jsonify({"status": "success"})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/kaggle_dataset/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/api/list_recordings', methods=['GET'])
def list_recordings():
    # Find all json metadata files
    files = []
    json_pattern = os.path.join(OUTPUT_FOLDER, "*.json")
    for meta_path in glob.glob(json_pattern):
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Find corresponding audio
            audio_filename = os.path.basename(meta_path).replace('.json', '.mp3')
            if os.path.exists(os.path.join(OUTPUT_FOLDER, audio_filename)):
                files.append({
                    "filename": audio_filename,
                    "metadata": metadata
                })
        except Exception as e:
            print(f"Error reading {meta_path}: {e}")
            
    # Sort by timestamp desc
    files.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
    return jsonify(files)

@app.route('/api/generate', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get('text')
    voice_name = data.get('voice_name', 'Leda') 
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        prompt = f"Please read the following text with the voice of {voice_name}. {data.get('system_instructions', '')} Text: {text}"
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
        )
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            # Inline audio is usually in the first part if successful
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    audio_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                    return jsonify({"audio_base64": audio_b64})
            
            return jsonify({"error": "No inline audio data received"}), 500
        else:
            return jsonify({"error": "No candidates returned"}), 500

    except Exception as e:
        print(f"Error generating audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save', methods=['POST'])
def save_audio():
    data = request.json
    audio_b64 = data.get('audio_base64')
    metadata = data.get('metadata', {}) 
    
    if not audio_b64:
        return jsonify({"error": "No audio data to save"}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        voice = metadata.get('voice', 'unknown')
        emotion = metadata.get('emotion', 'neutral')
        text_snippet = metadata.get('text', '')[:10].replace(" ", "_").replace("/", "")
        
        filename = f"{timestamp}_{voice}_{emotion}_{text_snippet}.mp3"
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        
        audio_bytes = base64.b64decode(audio_b64)
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
            
        metadata['timestamp'] = timestamp
        meta_filepath = filepath.replace('.mp3', '.json')
        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return jsonify({"message": "Saved successfully", "filename": filename})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)
