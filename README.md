# PsycheTTS: Emotional Text-to-Speech Research Platform

**PsycheTTS** is a comprehensive research project contributed to the **University of Sri Jayewardenepura**. It focuses on developing emotionally expressive Text-to-Speech models for Sinhala, utilizing advanced techniques like the Gemini 2.5 Pro TTS API for data collection and custom neural architectures for training.

## Project Structure

This repository is organized into two main modules:

### 1. `collector`
Tools for generating and curating high-quality emotional/neutral audio datasets using Google's Gemini API.

-   **`src/web/`**: A Flask-based web interface for:
    -   **Manual Helper Mode**: Streamlines the copy-paste workflow for AI Studio generation.
    -   **CSV Logging**: Automatically logs generated files to `dataset/metadata.csv`.
    -   **Dataset Explorer**: Review collected samples.
-   **`src/cli/`**: A Python CLI for batch generation (subject to API rate limits).
-   **`dataset/`**: The central repository for all collected `.wav` files and `metadata.csv`.

### 2. `research`
The core research codebase for training and experimenting with emotional TTS models.

-   **`train.py`**: Main training loop.
-   **`model.py`**: Neural network definitions (PsycheTTS architecture).
-   **`modules.py`**: Reusable blocks (Mamba, Style Encoders).
-   **`data_loader.py`**: Custom PyTorch dataset loaders.
-   **`psyche_tts_portal.html`**: Interactive research portal for visualizing results.

## Quick Start

### Setting up the Collector

1.  Navigate to `collector`:
    ```bash
    cd collector
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Web Helper:
    ```bash
    python src/web/server.py
    ```
4.  Open `http://localhost:8000/manual` to start collecting data.

### Setting up Research

1.  Navigate to `research`:
    ```bash
    cd research
    ```
2.  Install dependencies (PyTorch, etc.):
    ```bash
    pip install -r requirements.txt
    ```
3.  Start training:
    ```bash
    python train.py
    ```

## Collaboration Guidelines

-   **Dataset**: All new audio samples must be placed in `collector/dataset/` and logged in `collector/dataset/metadata.csv`.
-   **Modeling**: New model architectures should be added to `research/modules.py` and imported in `research/model.py`.

## License
MIT
