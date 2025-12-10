# <div align="center">PsycheTTS: Emotional Text-to-Speech Platform</div>

<div align="center">
  <h3>University of Sri Jayewardenepura Research Contribution</h3>
  <p>
    An advanced research initiative focused on developing high-fidelity, emotionally expressive Text-to-Speech models for low-resource languages (Sinhala).
  </p>
  
  [![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/Cortana-Devs/PsycheTTS-V2)
  [![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

  [View Research Portal](./research/psyche_tts_portal.html) • [Data Collector](./collector/README.md)
</div>

---

## 🚀 Project Overview

**PsycheTTS** combines the power of Large Language Models (Gemini 2.5 Pro) with efficient neural architectures (Bi-Directional Mamba) to solve the data scarcity problem in Sinhala TTS.

### Key Components

| Module | Description | Status |
| :--- | :--- | :--- |
| **Collector** | A Flask-based tool to generate & curate emotional audio samples using Gemini API. Features a **Manual Helper** for precise control. | ✅ **Verified Working** |
| **Research** | The core training pipeline implementing the **PsycheTTS Architecture** (Style Encoder + Mamba + Flow Matching). | 🚧 **In Development** |
| **Portal** | Interactive HTML dashboard for visualizing architecture, pipelines, and results. | ✨ **Live** |

---

## 📂 Directory Structure

```plaintext
PsycheTTS/
├── collector/          # Data Collection Suite
│   ├── dataset/        # Validated Audio Samples (.wav) & Metadata (.csv)
│   ├── src/web/        # Web Interface (Flask)
│   └── src/cli/        # Command Line Tools
│
├── research/           # Model Training & Experiments
│   ├── psyche_tts_portal.html  # Visualization Dashboard
│   ├── train.py        # Main Training Loop
│   ├── model.py        # Architecture Definitions
│   └── modules.py      # Core Building Blocks (Mamba, etc.)
```

---

## ⚡ Quick Start

### 1. Data Collector (Start Here)
Generate your own emotional dataset using our specialized tool.

```bash
# 1. Navigate to collector directory
cd collector

# 2. Install dependencies (First time only)
pip install -r requirements.txt

# 3. Launch the Server
python src/web/server.py
```

> **Note:** The server will start at `http://127.0.0.1:8000/manual`.
> Ensure you have your `GEMINI_API_KEY` ready if using automated features.

### 2. Research & Training
Train the model on the collected data.

```bash
# 1. Navigate to research directory
cd research

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Training
python train.py
```

---

## 🤝 Collaboration

This project is open for collaboration, specifically targeting researchers from the **University of Sri Jayewardenepura**.

*   **Dataset Contributions:** Please ensure all new samples are logged in `collector/dataset/metadata.csv`.
*   **Model Improvements:** Submit PRs for new modules in `research/modules.py`.

---

<div align="center">
  <sub>&copy; 2025 University of Sri Jayewardenepura & Cortana R&D Contribution. All Rights Reserved.</sub>
</div>
