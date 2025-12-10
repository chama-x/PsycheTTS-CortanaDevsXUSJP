# Red Team Report: PsycheTTS v3 Plan Review

## 1. Reality Checks (Hallucination Detection)

### A. The "CLAP" Assumption
**Plan Claim**: "CLAP outputs a sequence of Style Tokens ($S \in \mathbb{R}^{L \times D}$) capturing time-varying prosody."
**Reality Check**: ⚠️ **PARTIAL HALLUCINATION.**
-   Standard CLAP models (Laion/Microsoft) are designed for *global* classification (Sentence -> Single Vector). They are trained with Constrastive Loss on whole files.
-   **The Hole**: If you just take the final pooling layer of CLAP, you get a static vector, which brings you back to the "Global Emotion Vector" problem (just fancier).
-   **Fix**: You must extract the **Hidden States** (Last Transformer Layer) of the CLAP Audio Encoder *before* pooling. This preserves the time dimension ($L \times D$) needed for the "Style Stream".

### B. The "Zero-Shot" Physics Transfer
**Plan Claim**: "Model learns how to apply 'Anger' to /a:/ in English, so it will work for Sinhala /a:/."
**Reality Check**: ✅ **PLAUSIBLE but RISKY.**
-   **The Hole**: Prosody isn't just vowel quality. It's rhythm. English is Stress-Timed; Sinhala is Syllable-Timed. An "Angry" English rhythm superimposed on Sinhala words might sound like a "Drunken Foreigner" rather than an "Angry Local".
-   **Fix**: You need a **Duration Predictor** specifically fine-tuned on Sinhala. You cannot rely 100% on English rhythm. The plan mentions a "Rhythm Adapter" in risks—this needs to be a core *Requirement*, not a mitigation.

## 2. Holes in the Plan

### Hole 1: The Cross-Attention Explosion
**Critique**: "Injecting Cross-Attention into *every* Bi-Mamba block."
-   **Impact**: Mamba is fast because it's Recurrent ($O(N)$). Cross-Attention is $O(N \times S_{len})$. If you do this layer-wise, you destroy the inference speed advantage (45ms -> 200ms).
-   **Recommendation**: only inject Style at the **Bottleneck** (Middle Layer) or Input/Output. Do *not* put Attention inside every Mamba block.

### Hole 2: Text-to-Style Alignment
**Critique**: "User prompts 'Warm and friendly'."
-   **Impact**: CLAP text encoders are trained on captions like "A dog barking" or "Man speaking angrily". They may not understand abstract instructions like "Warm and friendly" well enough to guide subtle prosody.
-   **Recommendation**: Use a **Reference Audio Database**. It's safer to retrieve a "Warm Audio Sample" and use its embedding than to rely on CLAP's text understanding for nuanced adjectives.

## 3. Constructive Expert Review

### Positive Highlights
-   **The Core Pivot**: Moving from Global Vectors to **Cross-Attention** is exactly the right move to match Gemini/AudioLM's architecture without the cost.
-   **IPA Strategy**: Using IPA as the bridge is the only viable path for Low-Resource Sinhala. This is a solid design choice.

### Final "Green Light" Conditions
1.  **Modify Architecture**: Restriction style injection to the **Input Layer** or **Middle Bottleneck** only (preserve speed).
2.  **Mandate Rhythm Adapter**: Add a specific `SinhalaDurationPredictor` to the tasks.
3.  **Use Hidden States**: Explicitly specify using CLAP `hidden_states` in the implementation, not the `pooler_output`.

**Verdict**: The plan is **80% Solid**. Fix the CLAP pooling issue and the layer-wise attention cost, and it is ready to build.
