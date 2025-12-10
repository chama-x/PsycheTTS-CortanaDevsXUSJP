# PsycheTTS v3: "Authentic Identity" Implementation Plan
**Date:** December 10, 2025
**Status:** Planning
**Target:** SOTA Authentic Emotional TTS (Sinhala Zero-Shot) via Cross-Attentive Style Injection.

## 1. Executive Summary
To bridge the gap between PsycheTTS v2 (Fast/Robotic) and Gemini 2.5 (Slow/Authentic), we are moving to **PsycheTTS v3**.
We will replace the rigid "Global Emotion Vector" with a continuous **Style Stream** derived from **CLAP**. This allows the model to "hear" the desired style and modulate phonemes dynamically (Cross-Attention) rather than applying a blanket filter.

## 2. Technical Architecture (v3 Refinement)

### A. The "Style-Stream" Encoder (CLAP)
Instead of a text classification ("Happy"), we use a **Contrastive Audio-Text Encoder**.
-   **Model**: `Microsoft/CLAP` or `LAION-CLAP` (Distilled for mobile).
-   **Input**: Text Prompt ("Warm and friendly") OR Reference Audio (3s sample).
-   **Output**: A sequence of **Style Tokens** ($S \in \mathbb{R}^{L \times D}$) rather than a single vector.
-   **Benefit**: Captures time-varying prosody (e.g., "Start calm, end excited").

### B. Cross-Attentive Bi-Mamba Backbone
We inject style *inside* the depth of the model, not just at the start.
-   **Old (v2)**: `Mamba(Input + EmotionVector)`
-   **New (v3)**:
    ```python
    def forward(self, x, style_tokens):
        x = self.conv1d(x)          # Texture
        x = self.cross_attn(x, style_tokens) # <--- INJECTION POINT
        x = self.bi_mamba(x)        # Global Flow
        return x
    ```
-   **Mechanism**: The `CrossAttention` layer allows specific phonemes to "query" the style tokens. (e.g., "How should I pronounce this vowel given the 'Warm' style?").

### C. Training Strategy: "The Authenticity Transfer"
We cannot record 10,000 hours of Emotional Sinhala. We transfer "Physics".
1.  **Stage 1: The Physics of Emotion (90%)**
    -   Train on **English/Chinese** high-emotion datasets (ESD, RAVDESS).
    -   Input: **IPA Phonemes**.
    -   Model learns: *How to apply "Anger" to /a:/, /p/, /t/.*
2.  **Stage 2: The Sinhala Passenger (10%)**
    -   Eval on **Sinhala**.
    -   Since Sinhala is just a sequence of IPA phonemes (which the model knows), it applies the "Anger" physics perfectly.

## 3. Implementation Phases

### Phase 1: The Style Bridge (Foundation)
-   [ ] **Set up CLAP**: Implement a lightweight wrapper for the CLAP encoder using `transformers` or `onnx`.
-   [ ] **Projector**: Build a simple MLP to project CLAP embeddings ($D=768$) to our model dim ($D=512$).

### Phase 2: Core Architecture (The Brain)
-   [ ] **Modify `BiMambaBlock`**: Add `MultiheadAttention(query=x, key=style, value=style)`.
-   [ ] **Gated Fusion**: Ensure the style injection is gated (`tanh`) so it doesn't destabilize training initially.

### Phase 3: Training & Verification
-   [ ] **Loss Function**: Add `StyleReconstructionLoss` - can we reconstruct the style vector from the generated audio?
-   [ ] **Inference Loop**: Update `inference.py` to accept text/audio prompts for style.

## 4. Risks & Mitigations
-   **Risk**: Cross-Attention is $O(N^2)$.
    -   **Mitigation**: We only attend to a small number of *Style Tokens* (e.g., 8 or 16), not the full sequence. This keeps it linear-ish $O(N \times k)$.
-   **Risk**: Overfitting to English prosody.
    -   **Mitigation**: Use a localized "Rhythm Adapter" (Conv1d) for Sinhala duration modeling.

## 5. Verification Plan
-   **Unit Test**: generate audio from "Happy" text prompt. Compare spectrogram dynamic range vs v2.
-   **Acceptance Criteria**: Dynamic Range > 120Hz (matching Gemini).
