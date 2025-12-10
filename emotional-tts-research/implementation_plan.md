# PsycheTTS v3: "Authentic Identity" Implementation Plan
**Date:** December 10, 2025
**Status:** Planning
**Target:** SOTA Authentic Emotional TTS (Sinhala Zero-Shot) via Cross-Attentive Style Injection.

## 1. Executive Summary
To bridge the gap between PsycheTTS v2 (Fast/Robotic) and Gemini 2.5 (Slow/Authentic), we are moving to **PsycheTTS v3**.
We will replace the rigid "Global Emotion Vector" with a continuous **Style Stream** derived from **CLAP**. This allows the model to "hear" the desired style and modulate phonemes dynamically (Cross-Attention) rather than applying a blanket filter.

## 2. Technical Architecture (v3 Refinement)

### A. The "Style-Stream" Encoder (CLAP - Corrected)
-   **Model**: `Microsoft/CLAP` or `LAION-CLAP` (Distilled).
-   **CRITICAL FIX**: usage of **Hidden States** (Last Transformer Layer), NOT the pooled vector.
    -   *Why*: Pooling crushes time dynamics. Hidden states ($L \times D$) preserve the "Story of the Style" (e.g., rising intonation).
-   **Output**: A sequence of **Style Tokens** ($S \in \mathbb{R}^{L \times D}$).

### B. Cross-Attentive Bi-Mamba (Bottleneck Only)
To preserve our <100ms latency, we do **not** put attention in every block.
-   **Architecture**:
    1.  `Conv1d_Front` (Texture)
    2.  `BiMamba_Block_1` (Fast Context)
    3.  **`Style_Cross_Attention` (The Injection Bottleneck)**
    4.  `BiMamba_Block_2` (Fast Fusion)
-   **Mechanism**: The middle layer attends to the CLAP hidden states to bias the flow.

### C. The Rhythm Adapter (Sinhala Specific)
-   **Problem**: English rhythm (Stress-timed) $\neq$ Sinhala rhythm (Syllable-timed).
-   **Solution**: A lightweight **Duration Predictor** (Conv1d) trained *specifically* on the ~1h Sinhala CommonVoice dataset.
-   **Pipeline**: The core model handles *Pitch/Timbre* (Emotion), the Adapter handles *Timing*.

## 3. Implementation Phases

### Phase 1: The Style Bridge (Foundation)
-   [ ] **Set up CLAP**: Wrapper to extract `last_hidden_state`.
-   [ ] **Projector**: Linear projection from CLAP dim to Model dim.

### Phase 2: Core Architecture (Optimized)
-   [ ] **Modify `BiMambaBlock`**: Implement standard block.
-   [ ] **Create `BottleneckStyleBlock`**: The specialized Cross-Attentive layer.
-   [ ] **Implement `SinhalaDurationAdaptor`**: Simple stochastic duration predictor.

### Phase 3: Training & Verification
-   [ ] **Loss Function**: `StyleReconstructionLoss` + `DurationLoss`.
-   [ ] **Training**: Pre-train on English Emotion -> Fine-tune Duration on Sinhala.

## 4. Risks & Mitigations
-   **Risk**: CLAP text encoder is vague ("Warm").
    -   **Mitigation**: Support **Reference Audio** prompting as the primary high-fidelity control conditioning.
-   **Risk**: Overfitting to English prosody.
    -   **Mitigation**: The `SinhalaDurationAdaptor` is the dedicated firewall against this.

## 5. Verification Plan
-   **Unit Test**: generate audio from "Happy" text prompt. Compare spectrogram dynamic range vs v2.
-   **Acceptance Criteria**: Dynamic Range > 120Hz (matching Gemini) AND <100ms Latency.
