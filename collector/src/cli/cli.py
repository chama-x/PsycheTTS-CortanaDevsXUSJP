import os
import json
import base64
import time
import argparse
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load env but prioritize user provided key if needed
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAoDegs3eiBWtDki2jogPfJ5sZ5fY7HMP8")

OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'dataset')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load samples from utils
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils', 'sinhala_samples.json')

def generate_audio(client, text, voice, emotion, instructions):
    """
    Generates audio using Gemini 2.5 Pro Preview TTS.
    """
    # Construct a prompt that encapsulates the desired output
    # Note: As of the preview, specific vocal control might be via config or prompt engineering
    # We will use the prompt engineering approach combined with config if available.
    
    prompt = f"""
    Generate audio for the following text.
    Voice: {voice}
    Emotion: {emotion}
    Instructions: {instructions}
    
    Text: {text}
    """

    try:
        # Retry logic for 429
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash-exp', 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice
                                )
                            )
                        )
                    )
                )
                
                # Extract audio bytes from the first candidate's first part
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data:
                            return part.inline_data.data
                break # Success
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limited. Retrying in 60 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(60)
                else:
                    raise e
                    
        return None

    except Exception as e:
        print(f"Error generating audio for '{text[:10]}...': {e}")
        return None

def process_batch(limit=None):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    with open(SAMPLES_PATH, 'r') as f:
        samples = json.load(f)
        
    if limit:
        samples = samples[:limit]
        
    print(f"Starting batch generation for {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Generating: {sample['emotion']} - {sample['text'][:20]}...")
        
        audio_bytes = generate_audio(
            client, 
            sample['text'], 
            sample['voice'], 
            sample['emotion'], 
            sample['instructions']
        )
        
        if audio_bytes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{sample['voice']}_{sample['emotion']}_{i}.mp3"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            
            with open(filepath, "wb") as f:
                f.write(audio_bytes)
                
            # Save Metadata
            meta_filepath = filepath.replace('.mp3', '.json')
            sample['timestamp'] = timestamp
            sample['model'] = 'gemini-2.5-pro-preview-tts'
            with open(meta_filepath, "w") as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
                
            print(f"Saved to {filename}")
        else:
            print("Failed to generate.")
            
        time.sleep(20) # Rate limit politeness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Audio Batch Generator")
    parser.add_argument('--limit', type=int, help="Limit number of samples to process")
    args = parser.parse_args()
    
    process_batch(args.limit)
