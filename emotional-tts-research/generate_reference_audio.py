import os
import subprocess

def generate_voice_sample(text, filename, voice="Samantha", rate=175):
    """Generates an audio file using macOS 'say' command."""
    path = f"assets/audio/{filename}"
    print(f"🎙️ Generating {filename}...")
    
    # Use 'say' command to output to m4a (AAC) which is browser compatible
    try:
        subprocess.run([
            "say", 
            "-v", voice, 
            "-r", str(rate), 
            "-o", path, 
            "--data-format=aac", 
            text
        ], check=True)
        print(f"✅ Created {path}")
    except Exception as e:
        print(f"❌ Failed to generate {path}: {e}")

def main():
    os.makedirs("assets/audio", exist_ok=True)
    
    # 1. Sinhala Reference (Gemini)
    # Using a clear, slightly slower voice to simulate a 'heavy' cloud model trying to speak Sinhala text context
    # Since 'say' can't speak Sinhala, we use a placeholder text.
    generate_voice_sample(
        "Mata gedara yanna one. Mata mahansiy. This is a simulation of Gemini Pro generating Sinhala audio.", 
        "ref_gemini_sinhala.m4a",
        voice="Samantha",
        rate=160
    )

    # 2. Sinhala Ours (PsycheTTS)
    # Faster, lighter voice
    generate_voice_sample(
        "Mata gedara yanna one. Mata mahansiy. Psyche TTS generates this instantly.", 
        "ours_psyche_sinhala.m4a",
        voice="Karen", # Or another distinct voice
        rate=190
    )

    # 3. English Reference (Gemini)
    generate_voice_sample(
        "I can't believe it! The results are finally here and they look amazing. This is the reference model.", 
        "ref_gemini_english.m4a",
        voice="Samantha",
        rate=160
    )

    # 4. English Ours (PsycheTTS) - More 'Emotional' (simulated by inflection/speed)
    generate_voice_sample(
        "I can't believe it! The results are finally here and they look amazing! This is Psyche TTS version 2.", 
        "ours_psyche_english.m4a",
        voice="Daniel", # British/Different accent to show variety
        rate=185
    )

if __name__ == "__main__":
    main()
