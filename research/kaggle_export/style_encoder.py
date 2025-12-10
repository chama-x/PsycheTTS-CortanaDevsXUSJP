import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleEncoder(nn.Module):
    """
    PsycheTTS v3 Style Bridge.
    Wraps a frozen CLAP model to extract continuous 'Style Tokens' from text or audio.
    
    CRITICAL ARCHITECTURE DECISION (Red Team Fix):
    We use `last_hidden_state` (Sequence) instead of `pooler_output` (Vector).
    This preserves the time-dynamics of the style (e.g. rising intonation).
    """
    def __init__(self, device='cpu', hidden_dim=768, model_dim=512):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim # CLAP default
        self.model_dim = model_dim   # PsycheTTS dim
        
        print("⚡ Initializing CLAP Style Bridge (Microsoft/large)...")
        # In a real training run, we would load the pretrained model here:
        # self.clap = CLAPModel.from_pretrained("laion/clap-htsat-fused").to(device)
        # self.processor = CLAPProcessor.from_pretrained("laion/clap-htsat-fused")
        
        # For this implementation prototype, we mock the heavy backbone 
        # but implement the REAL projection logic.
        self.mock_backbone = True
        
        # The Projector: Maps CLAP space to Bi-Mamba space
        # We use a simple 2-layer MLP with Gelu to allow non-linear mapping
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, model_dim)
        )

    def forward_text(self, text_list):
        """
        Extracts style from text prompts (e.g. "Warm and friendly").
        Returns: Style Tokens [Batch, Seq, Model_Dim]
        """
        if self.mock_backbone:
            # Simulate CLAP hidden states for text
            # Batch size = len(text_list)
            # Seq len = ~10 tokens (short prompt)
            batch_size = len(text_list)
            seq_len = 12 
            
            # Random "Warm" embeddings
            mock_hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim).to(self.device)
        else:
            inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
            outputs = self.clap.get_text_features(**inputs, output_hidden_states=True)
            mock_hidden_states = outputs.hidden_states[-1] # Take last layer

        # PROJECT: CLAP -> Psyche
        style_tokens = self.projector(mock_hidden_states)
        return style_tokens

    def forward_audio(self, audio_tensor):
        """
        Extracts style from reference audio (Zero-Shot cloning).
        Returns: Style Tokens [Batch, Seq, Model_Dim]
        """
        if self.mock_backbone:
            # Simulate CLAP hidden states for audio
            # Audio embeddings are usually longer sequences
            batch_size = audio_tensor.shape[0]
            seq_len = 64 # Granular acoustic features
            
            mock_hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim).to(self.device)
        else:
            # inputs = self.processor(audios=audio_tensor, return_tensors="pt"... )
            # ...
            pass
            
        style_tokens = self.projector(mock_hidden_states)
        return style_tokens

if __name__ == "__main__":
    # Unit Test
    encoder = StyleEncoder()
    prompts = ["Warm and friendly", "Angry and fast"]
    
    tokens = encoder.forward_text(prompts)
    print(f"✅ Input Prompts: {len(prompts)}")
    print(f"✅ Output Shape: {tokens.shape} (Batch, Seq, Dim)")
    print("   -> Success: We have a SEQUENCE of tokens, not just a single vector.")
