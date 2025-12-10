import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiMambaBlock(nn.Module):
    """
    Bi-Directional Mamba Block with Conv1d Frontend (PsycheTTS v2).
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        self.norm = nn.LayerNorm(d_model)
        
        # 1. Convolutional Frontend
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=conv_kernel,
            groups=self.d_inner,
            padding=conv_kernel // 2,
        )
        self.act = nn.SiLU()

        # 2. SSM Parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def ssm_step(self, x):
        ssm_params = self.x_proj(x)
        dt, B, C = torch.split(ssm_params, [self.dt_rank, 16, 16], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y_ssm = x * F.sigmoid(dt) 
        return y_ssm

    def forward(self, x):
        input_res = x
        x = self.norm(x)
        
        x_and_res = self.in_proj(x)
        (x_in, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        
        x_conv = self.conv1d(x_in.transpose(1, 2)).transpose(1, 2)
        x_conv = x_conv[:, :x.shape[1], :] 
        x_conv = self.act(x_conv)
        
        # Bi-Directional SSM
        y_fwd = self.ssm_step(x_conv)
        x_bwd = torch.flip(x_conv, dims=[1])
        y_bwd = self.ssm_step(x_bwd)
        y_bwd = torch.flip(y_bwd, dims=[1])
        
        y = (y_fwd + y_bwd) * F.silu(res)
        out = self.out_proj(y)
        return out + input_res

class BottleneckStyleBlock(nn.Module):
    """
    PsycheTTS v3: Cross-Attentive Style Injector.
    Placed in the middle of the backbone to bias flow generation with CLAP style tokens.
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        # Gating mechanism (init near 0) to allow gradual learning
        self.gate = nn.Parameter(torch.tensor([0.0])) 

    def forward(self, x, style_tokens):
        """
        x: (Batch, Seq_Len, d_model) - The 'Content' Stream
        style_tokens: (Batch, Style_Len, d_model) - The 'Style' Stream (CLAP Hidden States)
        """
        residual = x
        x = self.norm(x)
        # Cross Attention: Query=Content, Key=Style, Value=Style
        attn_out, _ = self.attn(query=x, key=style_tokens, value=style_tokens)
        
        # Gated Residual Connection
        return residual + (torch.tanh(self.gate) * attn_out)

class SinhalaDurationAdaptor(nn.Module):
    """
    PsycheTTS v3: Rhythm/Duration Predictor.
    Trained specifically on Sinhala data to predict phoneme durations.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1) # Predict log-duration

    def forward(self, x):
        # x: (Batch, Seq, Dim)
        x = x.transpose(1, 2) # Conv expects (B, C, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.norm(x)
        log_dur = self.proj(x)
        return log_dur.squeeze(-1) # (Batch, Seq)

class MelDiscriminator(nn.Module):
    """
    PatchGAN-style Discriminator for Mel Spectrograms.
    Input: (B, 1, MelDim, Time)
    """
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (B, 1, 80, T)
            nn.Conv2d(in_channels, ndf, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x: (B, MelDim, Time) -> Unsqueeze to (B, 1, MelDim, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.net(x)

class Discriminator(nn.Module):
    """
    Wrapper for Discriminator Access
    """
    def __init__(self):
        super().__init__()
        self.mel_disc = MelDiscriminator()
        
    def forward(self, y, y_hat):
        # y, y_hat: (B, MelDim, Time)
        d_real = self.mel_disc(y)
        d_fake = self.mel_disc(y_hat)
        return [d_real], [d_fake], [], [] # Matching APIs for existing loss func (lists)


class TextEmotionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, emotion_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.emotion_proj = nn.Linear(emotion_dim, embed_dim)
        
    def forward(self, text_indices, emotion_vector):
        x = self.embedding(text_indices)
        e = self.emotion_proj(emotion_vector).unsqueeze(1)
        x = x + e 
        x = self.transformer(x)
        return x

class VectorFieldEstimator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels // 4),
            nn.SiLU(),
            nn.Linear(hidden_channels // 4, hidden_channels)
        )
        self.proj_in = nn.Linear(in_channels, hidden_channels)
        self.proj_out = nn.Linear(hidden_channels, in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )

    def forward(self, x, t, context):
        t_emb = self.time_mlp(t.unsqueeze(-1)).unsqueeze(1)
        x_emb = self.proj_in(x)
        h = x_emb + context + t_emb
        h = self.mlp(h)
        out = self.proj_out(h)
        return out
