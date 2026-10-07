"""
Audio Quality Enhancement Module
=================================
Spectral denoising and quality improvement for reconstructed/generated TTS audio.

Techniques implemented:
1. Spectral Gating (noise gate based on noise profile estimation)
2. Wiener Filtering (minimum mean-square error estimator)
3. Spectral Subtraction (classic noise reduction)
4. Post-processing (normalization, de-clicking, smoothing)
"""

import numpy as np
from scipy.signal import lfilter, medfilt
from scipy.ndimage import uniform_filter1d
import librosa


# ============================================================
# 1. SPECTRAL GATING
# ============================================================

def spectral_gate(audio, sr=22050, n_fft=2048, hop_length=256,
                  noise_duration=0.5, threshold_factor=1.5):
    """
    Noise reduction using spectral gating.
    
    Estimates the noise profile from the first `noise_duration` seconds
    of the audio (assuming it starts with silence/noise), then gates
    out frequency bins that fall below the noise threshold.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop length for STFT
        noise_duration: seconds of audio to use for noise estimation
        threshold_factor: multiplier for the noise threshold (higher = more aggressive)
    
    Returns:
        Denoised audio as 1D numpy array
    """
    # Compute STFT
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Estimate noise profile from the first N frames
    noise_frames = int(noise_duration * sr / hop_length)
    noise_frames = max(1, min(noise_frames, magnitude.shape[1]))
    
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Create a soft mask: suppress bins below noise threshold
    threshold = noise_profile * threshold_factor
    mask = np.maximum(magnitude - threshold, 0.0) / (magnitude + 1e-10)
    
    # Smooth the mask to avoid musical noise artifacts
    mask = uniform_filter1d(mask, size=3, axis=1)
    
    # Apply mask
    cleaned_magnitude = magnitude * mask
    
    # Reconstruct
    cleaned_D = cleaned_magnitude * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_D, hop_length=hop_length)
    
    return cleaned_audio


# ============================================================
# 2. WIENER FILTER
# ============================================================

def wiener_filter(audio, sr=22050, n_fft=2048, hop_length=256,
                  noise_duration=0.5):
    """
    Noise reduction using Wiener filtering.
    
    Estimates SNR per frequency bin and applies a gain that
    preserves high-SNR components while suppressing low-SNR ones.
    
    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop length for STFT
        noise_duration: seconds of audio to use for noise estimation
    
    Returns:
        Filtered audio as 1D numpy array
    """
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(D) ** 2
    
    # Estimate noise power spectrum
    noise_frames = int(noise_duration * sr / hop_length)
    noise_frames = max(1, min(noise_frames, power.shape[1]))
    noise_power = np.mean(power[:, :noise_frames], axis=1, keepdims=True)
    
    # Wiener gain: H = max(1 - noise/signal, 0)
    # This is the classic parametric Wiener filter
    gain = np.maximum(1.0 - noise_power / (power + 1e-10), 0.0)
    
    # Smooth gain to reduce musical noise
    gain = uniform_filter1d(gain, size=5, axis=1)
    
    # Apply gain to magnitude, keep phase
    cleaned_D = D * gain
    cleaned_audio = librosa.istft(cleaned_D, hop_length=hop_length)
    
    return cleaned_audio


# ============================================================
# 3. SPECTRAL SUBTRACTION
# ============================================================

def spectral_subtraction(audio, sr=22050, n_fft=2048, hop_length=256,
                         noise_duration=0.5, oversubtraction=2.0, floor=0.01):
    """
    Classic spectral subtraction for noise reduction.
    
    Subtracts the estimated noise magnitude spectrum from the signal,
    with an oversubtraction factor to be more aggressive and a
    spectral floor to prevent negative values from causing artifacts.
    
    Args:
        audio: 1D numpy array
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop length
        noise_duration: seconds for noise estimation
        oversubtraction: how aggressively to subtract (1.0 = exact, 2.0 = aggressive)
        floor: minimum spectral floor to prevent silence artifacts
    
    Returns:
        Enhanced audio as 1D numpy array
    """
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # Noise estimation
    noise_frames = int(noise_duration * sr / hop_length)
    noise_frames = max(1, min(noise_frames, magnitude.shape[1]))
    noise_mag = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Subtract with oversubtraction factor and spectral floor
    subtracted = magnitude - oversubtraction * noise_mag
    floored = np.maximum(subtracted, floor * magnitude)
    
    # Reconstruct
    cleaned_D = floored * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_D, hop_length=hop_length)
    
    return cleaned_audio


# ============================================================
# 4. POST-PROCESSING UTILITIES
# ============================================================

def normalize_audio(audio, target_peak=0.95):
    """Normalize audio to a target peak amplitude."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def remove_clicks(audio, kernel_size=3):
    """
    Remove click artifacts using median filtering.
    Clicks appear as sharp spikes in the waveform that median
    filtering naturally smooths out.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(audio, kernel_size=kernel_size).astype(np.float32)


def smooth_audio(audio, window_size=5):
    """Apply a simple moving average to smooth harsh transitions."""
    return uniform_filter1d(audio, size=window_size).astype(np.float32)


def trim_silence(audio, sr=22050, top_db=20):
    """Trim leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


# ============================================================
# 5. FULL ENHANCEMENT PIPELINE
# ============================================================

def enhance_audio(audio, sr=22050, n_fft=2048, hop_length=256,
                  method='spectral_gate', noise_duration=0.5,
                  do_trim=True, do_normalize=True, do_declick=True):
    """
    Full audio enhancement pipeline.
    
    Applies noise reduction followed by optional post-processing steps.
    
    Args:
        audio: 1D numpy array
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop length
        method: 'spectral_gate', 'wiener', 'spectral_subtraction', or 'all'
        noise_duration: seconds to estimate noise from
        do_trim: whether to trim silence
        do_normalize: whether to normalize output
        do_declick: whether to apply click removal
    
    Returns:
        Enhanced audio as 1D numpy array
    """
    enhanced = audio.copy()
    
    # Step 1: Noise reduction
    if method == 'spectral_gate':
        enhanced = spectral_gate(enhanced, sr, n_fft, hop_length, noise_duration)
    elif method == 'wiener':
        enhanced = wiener_filter(enhanced, sr, n_fft, hop_length, noise_duration)
    elif method == 'spectral_subtraction':
        enhanced = spectral_subtraction(enhanced, sr, n_fft, hop_length, noise_duration)
    elif method == 'all':
        # Chain: spectral subtraction → wiener → spectral gate (light)
        enhanced = spectral_subtraction(enhanced, sr, n_fft, hop_length, noise_duration,
                                         oversubtraction=1.5, floor=0.02)
        enhanced = wiener_filter(enhanced, sr, n_fft, hop_length, noise_duration)
        enhanced = spectral_gate(enhanced, sr, n_fft, hop_length, noise_duration,
                                  threshold_factor=1.0)
    
    # Step 2: Post-processing
    if do_declick:
        enhanced = remove_clicks(enhanced)
    
    if do_trim:
        enhanced = trim_silence(enhanced, sr)
    
    if do_normalize:
        enhanced = normalize_audio(enhanced)
    
    return enhanced


# ============================================================
# 6. SPECTROGRAM-LEVEL ENHANCEMENT
# ============================================================

def enhance_spectrogram(spectrogram, method='median', kernel_size=3):
    """
    Enhance a spectrogram matrix before converting to audio.
    This can improve Griffin-Lim reconstruction quality.
    
    Args:
        spectrogram: 2D numpy array (time_steps, freq_bins)
        method: 'median' for median filtering, 'smooth' for uniform smoothing
        kernel_size: filter kernel size
    
    Returns:
        Enhanced spectrogram as 2D numpy array
    """
    enhanced = spectrogram.copy()
    
    if method == 'median':
        # Apply median filter along the time axis per frequency bin
        for i in range(enhanced.shape[1]):
            enhanced[:, i] = medfilt(enhanced[:, i], kernel_size=kernel_size)
    elif method == 'smooth':
        # Smooth along time axis
        for i in range(enhanced.shape[1]):
            enhanced[:, i] = uniform_filter1d(enhanced[:, i], size=kernel_size)
    elif method == 'both':
        # First median (remove spikes), then smooth
        for i in range(enhanced.shape[1]):
            enhanced[:, i] = medfilt(enhanced[:, i], kernel_size=kernel_size)
            enhanced[:, i] = uniform_filter1d(enhanced[:, i], size=kernel_size)
    
    return enhanced
