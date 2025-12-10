# Gemini Audio Collector

A premium web interface to collect and curate high-quality audio samples from Gemini 2.5 Pro for TTS fine-tuning.

## Features
- **Premium UI**: Dark mode, glassmorphism, responsive design.
- **Configurable Generation**: Select Voice (Leda/Orus), Emotion, and provide System Instructions.
- **Dataset Export**: Saves Audio (.mp3) and Metadata (.json) to `kaggle_dataset` folder.

## Setup & Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**
   Make sure you have your Gemini API key in a `.env` file or export it:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

3. **Run the App**
   ```bash
   python src/web/server.py
   ```
   Open http://127.0.0.1:8000 in your browser.

4. **Collect Data**
   - Enter text (Sinhala or English).
   - Set tone/emotion.
   - Click **Generate**.
   - If satisfied, click **Save to Kaggle Dataset**.

## Output
Files are saved to `./dataset/` with the naming convention:
`YYYYMMDD_HHMMSS_{Voice}_{Emotion}_{TextSnippet}.mp3`
