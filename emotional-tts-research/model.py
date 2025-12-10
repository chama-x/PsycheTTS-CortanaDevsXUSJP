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

class PsycheTTS(nn.Module):
    """
    PsycheTTS v2: "Sweetspot" Architecture.
    
    1. Text/IPA Encoder: Transformer (Conditions on Emotion)
    2. Backbone: Conv-BiMamba Stack (Fast, Global+Local)
    3. Head: Consistency Flow Matching + Adversarial Training
    """
    def __init__(self, 
                 vocab_size=150, # Increased for IPA tokens
                 d_model=256, 
                 n_layers=6, 
                 emotion_dim=64, 
                 mel_dim=80):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Text & Emotion Encoder
        self.text_encoder = TextEmotionEncoder(vocab_size, d_model, emotion_dim)
        
        # 2. Main Backbone (Conv-BiMamba)
        self.backbone = nn.ModuleList([
            BiMambaBlock(d_model=d_model, conv_kernel=4) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # 3. Flow Matching Head
        self.vector_field = VectorFieldEstimator(in_channels=mel_dim, hidden_channels=d_model)

        # 4. Latent Projection
        self.mel_proj = nn.Linear(mel_dim, d_model)
        
    def forward_cfm(self, text_indices, emotion_vector, clean_mel, mask=None):
        """
        Standard Conditional Flow Matching Loss (Velocity Field MSE).
        Used for the primary 'guidance' of the vector field.
        """
        batch_size, mel_len, mel_dim = clean_mel.shape
        device = clean_mel.device
        
        cond = self.text_encoder(text_indices, emotion_vector) # (B, L_text, D)
        
        # FIX for Prototype: Match Text Length to Mel Length
        # In production, we use a Duration Predictor. Here, we interpolate.
        if cond.shape[1] != mel_len:
            cond = cond.transpose(1, 2) # (B, D, L_text)
            cond = F.interpolate(cond, size=mel_len, mode='linear', align_corners=False)
            cond = cond.transpose(1, 2) # (B, L_mel, D)

        # Time t ~ Uniform[0, 1]
        t = torch.rand(batch_size, device=device)
        
        x_0 = torch.randn_like(clean_mel)
        x_1 = clean_mel
        
        t_view = t.view(batch_size, 1, 1)
        x_t = (1 - t_view) * x_0 + t_view * x_1
        target_velocity = x_1 - x_0
        
        # Model Prediction
        x_t_hidden = self.mel_proj(x_t)
        h = x_t_hidden + cond
        for block in self.backbone:
            h = block(h)
        h = self.norm(h)
        
        pred_velocity = self.vector_field(x_t, t, h)
        
        # Velocity Loss
        loss = F.mse_loss(pred_velocity, target_velocity, reduction='none')
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            
        return loss.mean(), pred_velocity, x_t, t
    
    @torch.no_grad()
    def inference(self, text_indices, emotion_vector, n_steps=2):
        """
        Run inference with Euler ODE solver. 
        Targeting 2 steps thanks to Consistency/Adversarial training.
        """
        batch_size, seq_len = text_indices.shape
        device = text_indices.device
        
        cond = self.text_encoder(text_indices, emotion_vector)
        mel_dim = self.mel_proj.in_features
        x = torch.randn(batch_size, seq_len, mel_dim, device=device)
        
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i / n_steps
            t = torch.tensor([t_val] * batch_size, device=device)
            
            x_hidden = self.mel_proj(x)
            h = x_hidden + cond
            for block in self.backbone:
                h = block(h)
            h = self.norm(h)
            
            v_pred = self.vector_field(x, t, h)
            x = x + v_pred * dt
            
        return x
