import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'LJSpeech-1.1')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Audio parameters
    sample_rate = 22050
    n_fft = 2048
    hop_length = 256
    win_length = 1024
    spec_bins = 1025 # (n_fft // 2 + 1) -> 1025, as mentioned in paper
    preemphasis_k = 0.97
    
    # Text parameters
    use_phonemes = True
    
    # Model parameters (Seq2Seq with Attention)
    embed_dim = 256
    hidden_dim = 512 # As mentioned in paper: latent dimension is 512
    encoder_layers = 1
    decoder_layers = 1
    dropout = 0.1
