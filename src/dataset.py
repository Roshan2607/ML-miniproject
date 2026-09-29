import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from src.preprocessing import text_to_sequence
from src.config import Config

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, processed_dir):
        self.metadata = pd.read_csv(metadata_path, sep='|', header=None, 
                                    names=['id', 'transcription', 'normalized_transcription'])
        # Drop rows with missing text
        self.metadata.dropna(inplace=True)
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        text = row['normalized_transcription']
        audio_id = row['id']
        
        spec_path = os.path.join(self.processed_dir, f"{audio_id}.npy")
        if os.path.exists(spec_path):
            mel_spec = np.load(spec_path)
        else:
            # Fallback for missing processed files
            mel_spec = np.zeros((1, Config.spec_bins))
            
        seq = text_to_sequence(text)
        
        return torch.LongTensor(seq), torch.FloatTensor(mel_spec)

def collate_fn(batch):
    """Pad text sequences and spectrograms to max length (Zero padding as per paper)."""
    sequences, spectrograms = zip(*batch)
    
    seq_lengths = torch.LongTensor([len(seq) for seq in sequences])
    seq_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    spec_lengths = torch.LongTensor([spec.shape[0] for spec in spectrograms])
    spec_padded = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0.0)
    
    return seq_padded, seq_lengths, spec_padded, spec_lengths

def get_dataloader(metadata_path, processed_dir, batch_size=32, shuffle=True):
    dataset = LJSpeechDataset(metadata_path, processed_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
