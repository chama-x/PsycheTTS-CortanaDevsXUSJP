# Strategic Roadmap: Achieving Authentic Emotional Sinhala TTS
**Objective**: Replicate Gemini 2.5's "Warm & Authentic" prosody on mobile hardware (<100ms latency).

## 1. The Core Problem: "Global vs. Local" Control
Our analysis confirms why Gemini sounds better:
*   **Gemini (AudioLM)**: Treating audio as language allows it to "attend" to the style prompt ("Warm") at every millisecond. It can make a single vowel sound "warm" even if the word is "tired".
*   **PsycheTTS v1**: Uses a **Global Emotion Vector**. This acts like a single "filter" over the whole sentence. It cannot make complex, contradictory adjustments (like "Sad text" + "Happy tone").

## 2. The Solution: Cross-Attentive Bi-Mamba (The "Style-Stream" Architecture)
We will not switch to AudioLM (too slow). We will upgrade PsycheTTS to have a "Style Stream" that runs parallel to the text stream.

### A. The "Brain": CLAP Style Bridge
Instead of a simple text encoder (BERT), we use **CLAP (Contrastive Language-Audio Pretraining)**.
-   **Why?**: CLAP gives us a shared math space where "Warm voice" (Audio) and "Warm tone" (Text) equal the *same vector*.
-   **Function**: 
    1.  User types "speak warmly".
    2.  CLAP converts this to a dense **Style Embedding Sequence** (not just one vector, but a trajectory).

### B. The "Body": Cross-Attention Injection
We modify the `BiMambaBlock` core.
-   **Current**: `Input -> Conv1d -> Mamba -> Output`
-   **New**: `Input -> Conv1d -> [Cross-Attention to Style] -> Mamba -> Output`
-   **Effect**: As the model generates the Sinhala phoneme for "Home" (*gedara*), it "looks" at the CLAP style vector. The vector tells it: *"Stretch the 'e' vowel and raise pitch slightly to sound welcoming."*

### C. The "Voice": Adversarial Flow Matching (Remains)
We keep the v2 generator. It is the only technology capable of producing high-fidelity audio in 2 steps.

---

## 3. The Sinhala "Authenticity" Pipeline (Zero-Shot)

How do we get authentic Sinhala without a massive Sinhala studio dataset?

### Step 1: The "Universal" Training (English/Multi-Lingual)
*   We train the **Cross-Attentive Bi-Mamba** on massive *English, Chinese, and French* emotional datasets (ESD, RAVDESS).
*   **Input**: IPA Phonemes (Universal sound symbols).
*   **Task**: "Here is the IPA for 'Hello' -> Make it sound 'Angry'."

### Step 2: The Sinhala "Passenger" (Inference)
*   **Input**: Sinhala Text ("මට ගෙදර යන්න ඕනෙ") -> **IPA Converter** (`məʈə gedərə...`).
*   **Style**: "Warm and Friendly" -> **CLAP Encoder** -> Style Vectors.
*   **Result**: The model sees IPA tokens (which it knows) and Style Vectors (which it knows). It generates the audio using the *learned physics* of "Warmth" applied to the *phonetics* of Sinhala.
*   **Authenticity**: Comes from the model applying **high-fidelity acoustic rules** (breathing, pitch variation) perfectly to the Sinhala phonemes.

---

## 4. Execution Roadmap

### Phase 1: The Style Bridge (Week 1)
- [ ] Integrate **Laion-CLAP** (feature extractor).
- [ ] Freeze CLAP and build a projector to our latent dimension ($D=512$).

### Phase 2: The Surgeon (Week 2)
- [ ] Modify `modules.py`: Insert `MultiheadAttention` layer before the `BiMamba` mixer.
- [ ] Update `train.py`: Add Style Loss (Cosine similarity between Output Audio encoding and Input Style prompt).

### Phase 3: The Tuning (Week 3)
- [ ] Fine-tune on **CommonVoice Sinhala** (only 1 hour needed) just to perfect the *accent* (vowel transitions), relying on the base model for *emotion*.

## 5. Summary Verdict
To beat Gemini, we don't need their massive compute. We need **Precision Control**. By injecting style via **Cross-Attention**, PsycheTTS v3 will be able to "paint" emotion onto Sinhala phonemes with the same nuanced authenticity, but at **1/20th the cost**.
