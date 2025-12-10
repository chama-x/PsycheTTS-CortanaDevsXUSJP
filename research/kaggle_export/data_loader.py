import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from text_utils import TextProcessor

class EmotionalDataset(Dataset):
    """
    Handles loading of:
    1. Text (Converted to IPA indices)
    2. Emotion Vector (Pre-computed or Label ID)
    3. Mel Spectrogram (Target)
    """
    def __init__(self, data_list, text_processor, mel_dim=80):
        self.data_list = data_list
        self.tp = text_processor
        self.mel_dim = mel_dim

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # In real scenario: load audio from path, compute mel, text from metadata
        item = self.data_list[idx]
        text = item['text']
        emotion_vec = item['emotion_vec'] # Tensor (Dim,)
        mel = item['mel'] # Tensor (Time, MelDim)
        
        indices, _ = self.tp.process(text, lang=item.get('lang', 'en'))
        indices = torch.tensor(indices, dtype=torch.long)
        
        return {
            'text': indices,
            'emotion': emotion_vec,
            'mel': mel
        }

def collate_fn(batch):
    """
    Pads sequences to max length in batch.
    """
    texts = [item['text'] for item in batch]
    emotions = [item['emotion'] for item in batch]
    mels = [item['mel'] for item in batch]
    
    # Pad Text
    text_lens = [len(t) for t in texts]
    max_text_len = max(text_lens)
    padded_text = torch.zeros(len(texts), max_text_len, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_text[i, :len(t)] = t
        
    # Pad Mel
    mel_lens = [len(m) for m in mels]
    max_mel_len = max(mel_lens)
    mel_dim = mels[0].shape[-1]
    padded_mel = torch.zeros(len(mels), max_mel_len, mel_dim)
    mask = torch.zeros(len(mels), max_mel_len)
    
    for i, m in enumerate(mels):
        padded_mel[i, :len(m), :] = m
        mask[i, :len(m)] = 1
        
    return {
        'text': padded_text,
        'emotion': torch.stack(emotions),
        'mel': padded_mel,
        'mask': mask
    }

class SyntheticDataset(EmotionalDataset):
    """
    Generates random data for 'Overfit' testing.
    """
    def __init__(self, size=100, mel_dim=80, seq_len=100):
        tp = TextProcessor()
        data = []
        for _ in range(size):
            # Random English or Sinhala text
            if np.random.rand() > 0.5:
                text = "Hello world this is a test"
                lang = 'en'
            else:
                text = "මම ඔයාට ආදරෙයි"
                lang = 'si'
                
            em_vec = torch.randn(32) # Emotion Dim
            mel = torch.randn(seq_len, mel_dim)
            data.append({'text': text, 'emotion_vec': em_vec, 'mel': mel, 'lang': lang})
            
        super().__init__(data, tp, mel_dim)

if __name__ == "__main__":
    # Test Loader
    ds = SyntheticDataset(size=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    
    batch = next(iter(dl))
    print(f"Text Shape: {batch['text'].shape}")
    print(f"Mel Shape: {batch['mel'].shape}")
    print(f"Mask Shape: {batch['mask'].shape}")
