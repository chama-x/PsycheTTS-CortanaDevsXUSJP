import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import time
from tqdm import tqdm

# Import our architecture
from model import PsycheTTS, discriminator_loss, generator_loss
from modules import Discriminator
from data_loader import SyntheticDataset, collate_fn

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"💾 Saved Checkpoint: {path}")

def train_kaggle_quality():
    print("\n🚀 INITIALIZING PSYCHETTS v3: STUDIO QUALITY TRAINING RUN")
    print("=========================================================")
    
    # 1. Hardware Detection (Kaggle T4 x2 Support)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"✅ DUAL GPU DETECTED: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
            DISTRIBUTED = True
        else:
            print(f"✅ SINGLE GPU DETECTED: {torch.cuda.get_device_name(0)}")
            device = torch.device('cuda')
            DISTRIBUTED = False
    else:
        print("⚠️  WARNING: CPU/MPS MODE. TRAINING WILL BE SLOW.")
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        DISTRIBUTED = False

    # 2. Hyperparameters for "Authentic Quality"
    BATCH_SIZE = 16 * torch.cuda.device_count() if torch.cuda.is_available() else 4
    EPOCHS = 100  # Deep convergence
    LR = 1e-4
    GRAD_CLIP = 1.0
    
    print(f"⚙️ Config: Batch={BATCH_SIZE} | Epochs={EPOCHS} | AMP=True")

    # 3. Model Initialization
    # Increased dim/layers for "Studio Quality" capacity
    generator = PsycheTTS(
        vocab_size=160,    # IPA + Specials
        d_model=512,       # Wide backbone
        n_layers=8,        # Deep backbone
        emotion_dim=128,   
        mel_dim=80
    ).to(device)
    
    discriminator = Discriminator().to(device)

    # Multi-GPU Wrapping
    if DISTRIBUTED:
        print("🔗 Wrapping models in DataParallel...")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # 4. Data Loading (Simulated for Script, User must Plug Real Data)
    # Using larger size to simulate real epoch length
    print("📚 Loading Dataset...")
    ds = SyntheticDataset(size=1000, mel_dim=80, seq_len=400) 
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=2, pin_memory=True)

    # 5. Optimizers & Scaler
    g_opt = optim.AdamW(generator.parameters(), lr=LR, betas=(0.8, 0.99))
    d_opt = optim.AdamW(discriminator.parameters(), lr=LR, betas=(0.8, 0.99))
    scaler = GradScaler()

    # 6. Training Loop
    print("\n🔥 STARTING TRAINING...")
    for epoch in range(EPOCHS):
        generator.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        g_losses = []
        d_losses = []
        
        for batch in pbar:
            text = batch['text'].to(device)
            emotion = batch['emotion'].to(device)
            mel = batch['mel'].to(device)
            mask = batch['mask'].to(device)
            
            # --- STYLE INJECTION (Mock -> Real) ---
            # In real Kaggle run, replicate 'Warm' style for 50% batch to force learning
            style_prompts = ["Warm and Authentic"] * text.shape[0]

            # ==========================
            # 1. TRAIN DISCRIMINATOR
            # ==========================
            d_opt.zero_grad()
            
            with autocast():
                # Generate Fake (No Grad for Gen)
                # Since DataParallel wraps method, we access 'module' or call forward directly if wrapped
                # DataParallel doesn't expose custom methods easily. 
                # Workaround: Use forward pass or access module.
                if DISTRIBUTED:
                    fake_mel = generator.module.inference(text, emotion, n_steps=2, style_prompt_text=style_prompts)
                else:
                    fake_mel = generator.inference(text, emotion, n_steps=2, style_prompt_text=style_prompts)
                
                real_input = mel.transpose(1, 2)
                fake_input = fake_mel.detach().transpose(1, 2)
                
                d_real, d_fake, _, _ = discriminator(real_input, fake_input)
                loss_d, _, _ = discriminator_loss(d_real, d_fake)
            
            scaler.scale(loss_d).backward()
            scaler.step(d_opt)
            
            # ==========================
            # 2. TRAIN GENERATOR
            # ==========================
            g_opt.zero_grad()
            
            with autocast():
                # A. Flow Matching Loss
                # DataParallel automatically splits args, but keyword args can be tricky.
                # Passing vars positionally where possible or ensuring kwargs work.
                if DISTRIBUTED:
                    # forward_cfm handles the splitting
                    loss, pred_v, x_t, t = generator(text, emotion, mel, mask, style_prompt_text=style_prompts) 
                    # Note: We need to update forward() to map to forward_cfm in DataParallel usage 
                    # OR just call module.forward_cfm if inputs are manually split. 
                    # Standard DP splits input tensors dim0.
                    # For simplicity in this script, we assume generator.forward calls forward_cfm
                    fm_loss = loss.mean()
                else:
                    fm_loss, pred_v, x_t, t = generator.forward_cfm(text, emotion, mel, mask, style_prompt_text=style_prompts)

                # B. Adversarial Loss
                # Approximation: euler step to t=1
                x_1_approx = x_t + pred_v * (1.0 - t.view(-1, 1, 1))
                d_fake_g, _, _, _ = discriminator(real_input, x_1_approx.transpose(1, 2))
                loss_gen_adv, _ = generator_loss(d_fake_g)
                
                # Total G Loss (Weighted)
                loss_g = fm_loss + 0.1 * loss_gen_adv

            scaler.scale(loss_g).backward()
            
            # Gradient Clipping
            scaler.unscale_(g_opt)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), GRAD_CLIP)
            
            scaler.step(g_opt)
            scaler.update()
            
            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())
            
            pbar.set_postfix({'G_Loss': f"{loss_g.item():.3f}", 'D_Loss': f"{loss_d.item():.3f}"})

        # End of Epoch
        avg_g = sum(g_losses)/len(g_losses)
        avg_d = sum(d_losses)/len(d_losses)
        print(f"📉 Epoch {epoch+1} Summary: G_Loss={avg_g:.4f} | D_Loss={avg_d:.4f}")
        
        # Save Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/psychetts_v3_epoch_{epoch+1}.pt"
            state_dict = generator.module.state_dict() if DISTRIBUTED else generator.state_dict()
            torch.save(state_dict, save_path)
            print(f"✨ Saved: {save_path}")

    print("\n✅ TRAINING COMPLETE. READY FOR EXPORT.")

if __name__ == "__main__":
    train_kaggle_quality()
