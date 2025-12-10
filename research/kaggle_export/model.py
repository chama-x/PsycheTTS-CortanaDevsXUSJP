import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import BiMambaBlock, TextEmotionEncoder, VectorFieldEstimator

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l
        gen_losses.append(l.item())
    return loss, gen_losses

from style_encoder import StyleEncoder
from modules import BottleneckStyleBlock, SinhalaDurationAdaptor

class PsycheTTS(nn.Module):
    """
    PsycheTTS v3: "Authentic Identity" Architecture.
    
    1. Text/IPA Encoder: Transformer (Base Content)
    2. Style Bridge: CLAP -> Hidden States (Time-Varying Style)
    3. Backbone: Bi-Mamba with Bottleneck Cross-Attention
    4. Head: Consistency Flow Matching + Adversarial Training
    """
    def __init__(self, 
                 vocab_size=150, 
                 d_model=256, 
                 n_layers=6, 
                 emotion_dim=64, 
                 mel_dim=80):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Text & Emotion Encoder (Base Content)
        self.text_encoder = TextEmotionEncoder(vocab_size, d_model, emotion_dim)

        # 2. Style Bridge (v3 New)
        self.style_encoder = StyleEncoder(hidden_dim=768, model_dim=d_model)
        
        # 3. Backbone (Split for Bottleneck Injection)
        mid_layer = n_layers // 2
        self.backbone_front = nn.ModuleList([
            BiMambaBlock(d_model=d_model, conv_kernel=4) for _ in range(mid_layer)
        ])
        
        self.style_bottleneck = BottleneckStyleBlock(d_model=d_model)
        
        self.backbone_back = nn.ModuleList([
            BiMambaBlock(d_model=d_model, conv_kernel=4) for _ in range(n_layers - mid_layer)
        ])
        
        # 4. Rhythm Adapter (v3 New - Sinhala Specific)
        self.duration_adaptor = SinhalaDurationAdaptor(d_model=d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
        # 5. Flow Matching Head
        self.vector_field = VectorFieldEstimator(in_channels=mel_dim, hidden_channels=d_model)
        self.mel_proj = nn.Linear(mel_dim, d_model)
        
    def forward_cfm(self, text_indices, emotion_vector, clean_mel, mask=None, style_prompt_text=None):
        """
        Standard Conditional Flow Matching Loss + Duration & Style Aux Loss.
        """
        batch_size, mel_len, mel_dim = clean_mel.shape
        device = clean_mel.device

        # A. Content Condition
        cond = self.text_encoder(text_indices, emotion_vector) # (B, L_text, D)

        # B. Get Style Tokens (v3)
        if style_prompt_text:
            # List of strings, e.g. ["Warm and friendly"] * B
            style_tokens = self.style_encoder.forward_text(style_prompt_text)
        else:
            # Fallback (Neutral Style) - Mocking a zero-style token sequence
            style_tokens = torch.zeros(batch_size, 1, self.d_model, device=device)

        # C. Duration Prediction (Auxiliary)
        # In real training, we would compare this against specific aligner durations
        log_dur_pred = self.duration_adaptor(cond)

        # FIX: Interpolate Condition to Mel Length (for Flow Matching)
        if cond.shape[1] != mel_len:
            cond = cond.transpose(1, 2)
            cond = F.interpolate(cond, size=mel_len, mode='linear', align_corners=False)
            cond = cond.transpose(1, 2)

        # D. Flow Matching Setup
        t = torch.rand(batch_size, device=device)
        x_0 = torch.randn_like(clean_mel)
        x_1 = clean_mel
        
        t_view = t.view(batch_size, 1, 1)
        x_t = (1 - t_view) * x_0 + t_view * x_1
        target_velocity = x_1 - x_0
        
        # E. Backbone Flow with Injection
        x_t_hidden = self.mel_proj(x_t)
        h = x_t_hidden + cond
        
        # Front Blocks
        for block in self.backbone_front:
            h = block(h)
            
        # Bottleneck Injection
        h = self.style_bottleneck(h, style_tokens)
        
        # Back Blocks
        for block in self.backbone_back:
            h = block(h)
            
        h = self.norm(h)
        
        # F. Velocity Prediction
        pred_velocity = self.vector_field(x_t, t, h)
        
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            
        return loss.mean(), pred_velocity, x_t, t
    
    @torch.no_grad()
    def inference(self, text_indices, emotion_vector, n_steps=2, style_prompt_text=None):
        batch_size, seq_len = text_indices.shape
        device = text_indices.device
        
        # A. Content
        cond = self.text_encoder(text_indices, emotion_vector)
        
        # B. Style
        if style_prompt_text:
            style_tokens = self.style_encoder.forward_text(style_prompt_text)
        else:
            style_tokens = torch.zeros(batch_size, 1, self.d_model, device=device)
            
        # C. Predict Duration (Auto-Length) - Simplified for v3 prototype
        # log_dur = self.duration_adaptor(cond)
        # For now, we assume 1:1 mapping or external length control like v2
        
        mel_dim = self.mel_proj.in_features
        x = torch.randn(batch_size, seq_len, mel_dim, device=device) # Using seq_len as mel_len proxy
        
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i / n_steps
            t = torch.tensor([t_val] * batch_size, device=device)
            
            x_hidden = self.mel_proj(x)
            h = x_hidden + cond
            
            for block in self.backbone_front:
                h = block(h)
            
            h = self.style_bottleneck(h, style_tokens)
            
            for block in self.backbone_back:
                h = block(h)
                
            h = self.norm(h)
            
            v_pred = self.vector_field(x, t, h)
            x = x + v_pred * dt
            
        return x
