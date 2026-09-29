import os
import glob
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.preprocessing import audio_to_spectrogram

def main():
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    
    wavs_dir = os.path.join(Config.RAW_DATA_DIR, 'wavs')
    if not os.path.exists(wavs_dir):
        print(f"Error: Raw data directory {wavs_dir} not found.")
        print(f"Please place LJSpeech dataset in {Config.RAW_DATA_DIR}")
        return
        
    wav_files = glob.glob(os.path.join(wavs_dir, '*.wav'))
    print(f"Found {len(wav_files)} audio files. Starting preprocessing...")
    
    for wav_file in tqdm(wav_files):
        file_id = os.path.splitext(os.path.basename(wav_file))[0]
        
        try:
            spec = audio_to_spectrogram(
                wav_file, 
                n_fft=Config.n_fft, 
                hop_length=Config.hop_length, 
                win_length=Config.win_length,
                preemph=Config.preemphasis_k
            )
            
            out_path = os.path.join(Config.PROCESSED_DATA_DIR, f"{file_id}.npy")
            np.save(out_path, spec)
            
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

if __name__ == "__main__":
    main()
