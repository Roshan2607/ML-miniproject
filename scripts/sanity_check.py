"""
Sanity Check Script
====================
Verifies the entire pipeline works end-to-end with a tiny dummy dataset.
Run this before training to catch import errors, shape mismatches, etc.

Usage:
    python scripts/sanity_check.py
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.preprocessing import (
    text_to_sequence, sequence_to_text, get_vocab_size,
    preemphasis, inv_preemphasis, spectrogram_to_audio
)
from src.attention import Attention
from src.model import Seq2SeqTTS, Encoder, Decoder, SimpleNN
from src.enhancement import enhance_audio, enhance_spectrogram
from src.dataset import collate_fn


def check(name, condition, detail=""):
    status = "✓ PASS" if condition else "✗ FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


def main():
    print("=" * 60)
    print("SANITY CHECK — End-to-End TTS Pipeline")
    print("=" * 60)
    passed = 0
    total = 0

    # --- 1. Text Preprocessing ---
    print("\n1. Text Preprocessing")
    text = "Hello world"
    seq = text_to_sequence(text)
    total += 1
    if check("text_to_sequence", len(seq) > 0, f"'{text}' → {len(seq)} tokens"):
        passed += 1

    vocab_size = get_vocab_size()
    total += 1
    if check("get_vocab_size", vocab_size > 0, f"vocab_size = {vocab_size}"):
        passed += 1

    symbols = sequence_to_text(seq)
    total += 1
    if check("sequence_to_text", len(symbols) > 0, f"→ {symbols[:5]}..."):
        passed += 1

    # --- 2. Audio Preprocessing ---
    print("\n2. Audio Preprocessing (FIR Filter)")
    dummy_wav = np.random.randn(22050).astype(np.float32)  # 1 second
    filtered = preemphasis(dummy_wav, k=Config.preemphasis_k)
    total += 1
    if check("preemphasis", filtered.shape == dummy_wav.shape):
        passed += 1

    recovered = inv_preemphasis(filtered, k=Config.preemphasis_k)
    total += 1
    if check("inv_preemphasis", np.allclose(dummy_wav, recovered, atol=1e-5)):
        passed += 1

    # --- 3. Dummy Spectrogram ---
    print("\n3. Spectrogram Pipeline")
    dummy_spec = np.random.rand(50, Config.spec_bins).astype(np.float32)
    audio_out = spectrogram_to_audio(dummy_spec)
    total += 1
    if check("spectrogram_to_audio", len(audio_out) > 0, f"→ {len(audio_out)} samples"):
        passed += 1

    # --- 4. Collate / Zero Padding ---
    print("\n4. Zero Padding (collate_fn)")
    seq1 = torch.LongTensor(text_to_sequence("Short"))
    seq2 = torch.LongTensor(text_to_sequence("A much longer sentence here"))
    spec1 = torch.randn(30, Config.spec_bins)
    spec2 = torch.randn(80, Config.spec_bins)

    batch = [(seq1, spec1), (seq2, spec2)]
    seq_padded, seq_lens, spec_padded, spec_lens = collate_fn(batch)
    total += 1
    if check("collate_fn shapes",
             seq_padded.shape[0] == 2 and spec_padded.shape[0] == 2,
             f"text: {seq_padded.shape}, spec: {spec_padded.shape}"):
        passed += 1

    total += 1
    if check("padding lengths correct",
             seq_lens[0] != seq_lens[1] and spec_lens[0] != spec_lens[1]):
        passed += 1

    # --- 5. Model Architecture ---
    print("\n5. Model Architecture")
    device = torch.device('cpu')

    # Encoder
    encoder = Encoder(vocab_size, Config.embed_dim, Config.hidden_dim).to(device)
    enc_out, h, c = encoder(seq_padded, seq_lens)
    total += 1
    if check("Encoder forward",
             enc_out.shape[2] == Config.hidden_dim * 2,
             f"output: {enc_out.shape}"):
        passed += 1

    # Attention
    attn = Attention(Config.hidden_dim).to(device)
    context, alphas = attn(h, enc_out)
    total += 1
    if check("Attention forward",
             alphas.shape[1] == seq_padded.shape[1],
             f"alphas: {alphas.shape}"):
        passed += 1

    # Full Seq2Seq
    model = Seq2SeqTTS(vocab_size, Config.embed_dim, Config.hidden_dim, Config.spec_bins).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    total += 1
    if check("Seq2SeqTTS init", param_count > 0, f"{param_count:,} parameters"):
        passed += 1

    output = model(seq_padded, seq_lens, spec_padded, teacher_forcing_ratio=0.5)
    total += 1
    if check("Seq2SeqTTS forward",
             output.shape == spec_padded.shape,
             f"output: {output.shape}"):
        passed += 1

    # SimpleNN
    max_seq_len = seq_padded.shape[1]
    target_len = spec_padded.shape[1]
    simple_nn = SimpleNN(vocab_size, Config.embed_dim, max_seq_len, target_len, Config.spec_bins).to(device)
    nn_out = simple_nn(seq_padded)
    total += 1
    if check("SimpleNN forward",
             nn_out.shape == (2, target_len, Config.spec_bins),
             f"output: {nn_out.shape}"):
        passed += 1

    # --- 6. Loss & Backprop ---
    print("\n6. Loss & Backpropagation")
    loss = torch.nn.functional.mse_loss(output, spec_padded)
    loss.backward()
    total += 1
    if check("backward pass", True, f"loss = {loss.item():.4f}"):
        passed += 1

    # --- 7. Enhancement ---
    print("\n7. Audio Enhancement")
    test_audio = np.random.randn(22050).astype(np.float32) * 0.1

    enhanced_sg = enhance_audio(test_audio, method='spectral_gate')
    total += 1
    if check("spectral_gate", len(enhanced_sg) > 0):
        passed += 1

    enhanced_wf = enhance_audio(test_audio, method='wiener')
    total += 1
    if check("wiener_filter", len(enhanced_wf) > 0):
        passed += 1

    enhanced_ss = enhance_audio(test_audio, method='spectral_subtraction')
    total += 1
    if check("spectral_subtraction", len(enhanced_ss) > 0):
        passed += 1

    enhanced_all = enhance_audio(test_audio, method='all')
    total += 1
    if check("full_pipeline", len(enhanced_all) > 0):
        passed += 1

    enh_spec = enhance_spectrogram(dummy_spec, method='both')
    total += 1
    if check("enhance_spectrogram", enh_spec.shape == dummy_spec.shape):
        passed += 1

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! Pipeline is ready for training.")
    else:
        print(f"WARNING: {total - passed} check(s) failed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
