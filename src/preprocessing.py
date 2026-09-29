import os
import numpy as np
import librosa
from scipy.signal import lfilter
from g2p_en import G2p

g2p = G2p()

# Phoneme vocabulary including CMU dict phonemes, alphabets and special tokens
valid_symbols = [
    'PAD', 'EOS', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
    't', 'u', 'v', 'w', 'x', 'y', 'z', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 
    'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 
    'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 
    'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 
    'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 
    'UH0', 'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]
_symbol_to_id = {s: i for i, s in enumerate(valid_symbols)}
_id_to_symbol = {i: s for i, s in enumerate(valid_symbols)}

def get_vocab_size():
    return len(valid_symbols)

def text_to_sequence(text):
    """Converts text to a sequence of phoneme/char IDs using CMU dict mapping."""
    phonemes = g2p(text)
    sequence = []
    for p in phonemes:
        if p in _symbol_to_id:
            sequence.append(_symbol_to_id[p])
    sequence.append(_symbol_to_id['EOS'])
    return sequence

def sequence_to_text(sequence):
    """Converts a sequence of IDs back to text/phonemes."""
    return [_id_to_symbol[id] for id in sequence if id in _id_to_symbol]

def preemphasis(wav, k=0.97):
    """FIR filter to smooth noise as described in paper."""
    return lfilter([1, -k], [1], wav)

def inv_preemphasis(wav, k=0.97):
    """Inverse FIR filter."""
    return lfilter([1], [1, -k], wav)

def audio_to_spectrogram(audio_path, n_fft=2048, hop_length=256, win_length=1024, preemph=0.97):
    """
    FIR filter -> STFT -> Spectrogram
    Output shape: (time_steps, 1025)
    """
    y, sr = librosa.load(audio_path, sr=None)
    y = preemphasis(y, preemph)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)
    return np.log1p(mag).T # (time_steps, 1025)

def spectrogram_to_audio(spectrogram, n_fft=2048, hop_length=256, win_length=1024, preemph=0.97):
    """
    Spectrogram -> Inverse STFT -> Inverse FIR filter
    """
    mag = np.expm1(spectrogram.T)
    y = librosa.griffinlim(mag, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    y = inv_preemphasis(y, preemph)
    return y
