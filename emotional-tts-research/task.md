# Task: PsycheTTS v3 - "Authentic Style" Implementation
**Objective**: achieve "Gemini-Level" authenticity in Sinhala TTS by upgrading the architecture to **Cross-Attentive Bi-Mamba** with **CLAP-based Style Injection**, maintaining <100ms latency.

- [ ] **Planning Phase (v3)** <!-- id: 100 -->
    - [/] Define v3 Architecture (CLAP + Cross-Attention) <!-- id: 101 -->
    - [ ] Create Implementation Plan Artifact (v3) <!-- id: 102 -->
    - [ ] User Review of v3 Plan <!-- id: 103 -->

- [ ] **Phase 1: The Style Bridge (CLAP)** <!-- id: 104 -->
    - [ ] Integrate Pre-trained CLAP (Laion/Microsoft) <!-- id: 105 -->
    - [ ] Implement `StyleEncoder` module (Projection to $D_{model}$) <!-- id: 106 -->
    - [ ] Create "Style Token" Extraction pipeline <!-- id: 107 -->

- [ ] **Phase 2: Core Architecture Upgrade** <!-- id: 108 -->
    - [ ] Implement `CrossAttentionBlock` (Lightweight) <!-- id: 109 -->
    - [ ] Modify `BiMambaBlock` to accept `style_context` <!-- id: 110 -->
    - [ ] Update `PsycheTTS` forward pass for `(text, style_audio)` input <!-- id: 111 -->

- [ ] **Phase 3: Training Pipeline** <!-- id: 112 -->
    - [ ] Add `StyleConsistencyLoss` (Cosine Sim) <!-- id: 113 -->
    - [ ] Update Dataloader for Reference Audio pairing <!-- id: 114 -->
    - [ ] Training Run: Zero-Shot Transfer (English Data -> Sinhala Eval) <!-- id: 115 -->

- [x] **Phase 4: Research Portal & Visualization** <!-- id: 21 -->
    - [x] Design "The Wall", "The Process", "Horizon" narratives <!-- id: 22 -->
    - [x] Build Audio Comparison benchmarks <!-- id: 23 -->
    - [x] Launch `psyche_tts_portal.html` <!-- id: 26 -->
