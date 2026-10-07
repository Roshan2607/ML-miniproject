"""
Audio Enhancement Demo Script
==============================
Demonstrates the enhancement pipeline on reconstructed TTS audio.

Usage:
    python scripts/enhance_demo.py --input <path_to_wav> --output <output_dir>
    python scripts/enhance_demo.py --from-spectrogram <path_to_npy> --output <output_dir>
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.preprocessing import spectrogram_to_audio, audio_to_spectrogram
from src.enhancement import (
    enhance_audio,
    enhance_spectrogram,
    normalize_audio,
    spectral_gate,
    wiener_filter,
    spectral_subtraction
)


def enhance_from_wav(input_path, output_dir):
    """Load a WAV file, apply all enhancement methods, save comparisons."""
    print(f"\nLoading audio: {input_path}")
    audio, sr = sf.read(input_path)
    
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # mono
    
    print(f"  Duration: {len(audio)/sr:.2f}s | Sample rate: {sr} Hz")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original (normalized)
    original = normalize_audio(audio.copy())
    sf.write(os.path.join(output_dir, "0_original.wav"), original, sr)
    
    # Method 1: Spectral Gating
    print("  Applying Spectral Gating...")
    sg = enhance_audio(audio, sr, method='spectral_gate')
    sf.write(os.path.join(output_dir, "1_spectral_gate.wav"), sg, sr)
    
    # Method 2: Wiener Filter
    print("  Applying Wiener Filter...")
    wf = enhance_audio(audio, sr, method='wiener')
    sf.write(os.path.join(output_dir, "2_wiener_filter.wav"), wf, sr)
    
    # Method 3: Spectral Subtraction
    print("  Applying Spectral Subtraction...")
    ss = enhance_audio(audio, sr, method='spectral_subtraction')
    sf.write(os.path.join(output_dir, "3_spectral_subtraction.wav"), ss, sr)
    
    # Method 4: Full pipeline (all chained)
    print("  Applying Full Enhancement Pipeline...")
    full = enhance_audio(audio, sr, method='all')
    sf.write(os.path.join(output_dir, "4_full_pipeline.wav"), full, sr)
    
    print(f"\n{'='*50}")
    print(f"All enhanced files saved to: {output_dir}")
    print(f"{'='*50}")
    print(f"  0_original.wav            - Original audio (normalized)")
    print(f"  1_spectral_gate.wav       - Spectral Gating denoising")
    print(f"  2_wiener_filter.wav        - Wiener Filter denoising")
    print(f"  3_spectral_subtraction.wav - Spectral Subtraction denoising")
    print(f"  4_full_pipeline.wav        - All methods chained together")


def enhance_from_spectrogram(spec_path, output_dir):
    """Load a spectrogram .npy, enhance it, reconstruct, and save."""
    print(f"\nLoading spectrogram: {spec_path}")
    spectrogram = np.load(spec_path)
    print(f"  Shape: {spectrogram.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    sr = Config.sample_rate
    
    # A) Direct reconstruction (no enhancement)
    print("  Reconstructing without enhancement...")
    audio_raw = spectrogram_to_audio(spectrogram)
    audio_raw = normalize_audio(audio_raw)
    sf.write(os.path.join(output_dir, "0_raw_reconstruction.wav"), audio_raw, sr)
    
    # B) Spectrogram-level enhancement BEFORE reconstruction
    print("  Enhancing spectrogram (median filter) then reconstructing...")
    spec_enhanced = enhance_spectrogram(spectrogram, method='median', kernel_size=3)
    audio_spec_enh = spectrogram_to_audio(spec_enhanced)
    audio_spec_enh = normalize_audio(audio_spec_enh)
    sf.write(os.path.join(output_dir, "1_spectrogram_enhanced.wav"), audio_spec_enh, sr)
    
    # C) Spectrogram smoothing + reconstruction
    print("  Enhancing spectrogram (smooth + median) then reconstructing...")
    spec_both = enhance_spectrogram(spectrogram, method='both', kernel_size=5)
    audio_spec_both = spectrogram_to_audio(spec_both)
    audio_spec_both = normalize_audio(audio_spec_both)
    sf.write(os.path.join(output_dir, "2_spectrogram_smoothed.wav"), audio_spec_both, sr)
    
    # D) Raw reconstruction + audio-level spectral gating
    print("  Reconstructing then applying Spectral Gating...")
    audio_sg = enhance_audio(audio_raw, sr, method='spectral_gate')
    sf.write(os.path.join(output_dir, "3_audio_spectral_gate.wav"), audio_sg, sr)
    
    # E) Combined: spectrogram enhancement + audio enhancement
    print("  Full pipeline: spectrogram enhancement + audio enhancement...")
    spec_pre = enhance_spectrogram(spectrogram, method='both', kernel_size=3)
    audio_combo = spectrogram_to_audio(spec_pre)
    audio_combo = enhance_audio(audio_combo, sr, method='all')
    sf.write(os.path.join(output_dir, "4_full_pipeline.wav"), audio_combo, sr)
    
    print(f"\n{'='*60}")
    print(f"All enhanced files saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"  0_raw_reconstruction.wav   - Direct Griffin-Lim (no enhancement)")
    print(f"  1_spectrogram_enhanced.wav - Median filter on spectrogram")
    print(f"  2_spectrogram_smoothed.wav - Median + smoothing on spectrogram")
    print(f"  3_audio_spectral_gate.wav  - Spectral gating on audio")
    print(f"  4_full_pipeline.wav        - Both spectrogram + audio enhancement")


def main():
    parser = argparse.ArgumentParser(description="Audio Enhancement Demo")
    parser.add_argument('--input', type=str, help="Path to input WAV file")
    parser.add_argument('--from-spectrogram', type=str, help="Path to input .npy spectrogram")
    parser.add_argument('--output', type=str, default='outputs/samples',
                        help="Output directory for enhanced files")
    
    args = parser.parse_args()
    
    if args.from_spectrogram:
        enhance_from_spectrogram(args.from_spectrogram, args.output)
    elif args.input:
        enhance_from_wav(args.input, args.output)
    else:
        # Default: pick the first processed spectrogram if available
        processed = Config.PROCESSED_DATA_DIR
        if os.path.exists(processed):
            npys = [f for f in os.listdir(processed) if f.endswith('.npy')]
            if npys:
                spec_path = os.path.join(processed, npys[0])
                print(f"No input specified. Using first available spectrogram: {spec_path}")
                enhance_from_spectrogram(spec_path, args.output)
                return
        
        print("Usage:")
        print("  python scripts/enhance_demo.py --input audio.wav")
        print("  python scripts/enhance_demo.py --from-spectrogram spec.npy")


if __name__ == "__main__":
    main()
