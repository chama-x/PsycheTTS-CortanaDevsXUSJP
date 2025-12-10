# <div align="center">Gemini Audio Collector</div>

<div align="center">
  <h3>High-Fidelity Training Data Generator</h3>
  <p>
    A premium web interface to collect and curate high-quality audio samples from Gemini 2.5 Pro for TTS fine-tuning.
  </p>

  [![Status](https://img.shields.io/badge/Module-Collector-blue?style=for-the-badge)](../)
  [![Tech](https://img.shields.io/badge/Tech-Flask_Web_App-green?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
</div>

---

## ✨ Features

- **Premium UI**: Dark mode, glassmorphism aesthetics, and fully responsive design.
- **Configurable Generation**: Granular control over Voice (Leda/Orus), Emotion, and System Instructions.
- **Manual Helper Mode**: Specialized interface to streamline the copy-paste workflow from Google AI Studio.
- **Automated Logging**: Seamlessly saves audio (.wav) and syncs metadata to `dataset/metadata.csv`.

---

## 🛠️ Setup & Usage

### 1. Installation

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 2. Configuration (Optional)
If using the automated features (future implementation), set your API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 3. Launch Application

```bash
python src/web/server.py
```

> **Success!** The server is now running at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📝 Collection Workflow

1.  **Open Interface**: Go to `http://127.0.0.1:8000/manual`.
2.  **Generate Audio**: Use Google AI Studio to generate speech.
3.  **Input Details**: Paste the Sinhala text and select the corresponding emotion/voice in the local UI.
4.  **Save**: Click **"Save to Dataset"**.
    *   *Result:* Audio is saved to `./dataset/` and an entry is added to `metadata.csv`.

---

## 📂 Output Structure

Files are strictly organized for the `Research` module's data loader.

- **Storage**: `./dataset/`
- **Naming Convention**: `YYYYMMDD_HHMMSS_{Voice}_{Emotion}_{TextSnippet}.wav`
- **Metadata**: `id | filename | text | emotion | voice | timestamp`

---

<div align="center">
  <sub>Part of the PsycheTTS Research Platform</sub>
</div>
