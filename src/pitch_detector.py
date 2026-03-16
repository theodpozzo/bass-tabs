"""
Pitch extraction module for bass audio.

Analyzes an audio waveform, processes it through a neural network, 
and converts it into a discrete sequence of musical notes with timestamps.

Pipeline Architecture:
----------------------
1. Audio Load & Trim  (librosa)
2. Pitch Detection    (SwiftF0 - Lightweight Deep Learning)
3. Median Filtering   (Smooths 1-frame pitch anomalies)
4. MIDI Quantization  (Snaps raw frequencies to the 12-tone grid)
5. Note Segmentation  (Groups identical consecutive frames)
6. Duration Filtering (Removes micro-transients and finger noise)
"""

import librosa
import numpy as np
from scipy.signal import medfilt
from swift_f0 import *  # Modern, lightweight replacement for CREPE

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Visualization & Debugging
# ---------------------------------------------------------------------


def plot_pitch_data(times, raw_hz, smoothed_hz, quantized_midi, confidences):
    """
    Generates a two-panel Matplotlib graph to visually debug the pitch pipeline.
    The script execution will pause until the user closes the graph window.
    """
    print("📊 Generating pitch visualization graph...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- TOP PLOT: Frequency Tracking & Quantization ---
    # 1. Raw neural network output (noisy, contains vibrato and bends)
    ax1.plot(times, raw_hz, label='Raw SwiftF0 (Hz)',
             color='orange', linewidth=2)

    # 2. Post-median filter output (smooths out rapid 1-frame jumps)
    ax1.plot(times, smoothed_hz, label='Smoothed (Median Filter)',
             color='blue', linewidth=2)

    # 3. Quantized MIDI grid
    # Converts MIDI integers back to Hz to overlay on the same axis, showing the "stair-step" snap.
    quantized_hz = librosa.midi_to_hz(quantized_midi)
    ax1.plot(times, quantized_hz, label='Quantized MIDI (Hz)',
             color='red', linewidth=2)

    ax1.set_title("Pitch Tracking: Raw Hz vs. Quantized MIDI Grid")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_ylim(20, 400)  # Locked to standard bass guitar frequencies
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- BOTTOM PLOT: Voicing Confidence & Thresholds ---
    # Plots the model's certainty that a pitched note is actually playing (vs. noise/silence)
    ax2.plot(times, confidences, label='SwiftF0 Confidence', color='green')

    # Visual marker for the strict 0.9 confidence threshold set in the SwiftF0 initialization
    ax2.axhline(0.9, color='red', linestyle='--',
                label='Gating Threshold (0.9)')

    # Shaded regions indicate frames that pass the 0.9 confidence check
    ax2.fill_between(times, 0, confidences, where=(confidences > 0.9),
                     color='green', alpha=0.2, label='Accepted Voicing')

    ax2.set_title("Voicing Confidence (Silence Filtering)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Confidence (0 to 1)")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Main Pitch Extraction Pipeline
# ---------------------------------------------------------------------


def extract_notes(audio_path: str) -> list[dict]:
    """
    Detects discrete notes played in an isolated bass recording.

    Parameters:
        audio_path (str): Filepath to the isolated audio stem.

    Returns:
        list[dict]: A chronological sequence of note events.
                    Example: [{"note": "E2", "start_time": 1.20, "duration": 0.45}]
    """
    # -----------------------------------------------------------------
    # 1. Audio Loading & Preprocessing
    # -----------------------------------------------------------------
    # Load at 16kHz (SwiftF0's native training rate) and collapse to mono.
    audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    # Amplitude Gating: Trims leading and trailing silence.
    # top_db=25 removes any audio at the start/end that is 25dB quieter than the peak volume.
    audio_data, _ = librosa.effects.trim(audio_data, top_db=25)

    # -----------------------------------------------------------------
    # 2. Pitch Detection (Neural Network)
    # -----------------------------------------------------------------
    # SwiftF0 is a ~95k parameter ONNX model optimizing speed and noise resistance.
    # fmax=400.0 caps detection to the upper bass range to ignore high-pitched artifacts.
    # confidence_threshold=0.9 applies strict internal gating for voiced frames.
    pitch_estimator = SwiftF0(
        fmin=SwiftF0.DEFAULT_FMIN, fmax=400.0, confidence_threshold=0.9)

    result = pitch_estimator.detect_from_array(audio_data, sample_rate)

    # Convert native outputs to 1D Numpy arrays for mathematical operations
    result.timestamps = np.array(result.timestamps)
    result.pitch_hz = np.array(result.pitch_hz)
    result.confidence = np.array(result.confidence)

    # Debug: Print sample frame data
    print(
        f"Example frame: Time={result.timestamps[100]:.2f}s, Freq={result.pitch_hz[100]:.2f}Hz, Conf={result.confidence[100]:.2f}")

    # -----------------------------------------------------------------
    # 3. Smoothing & Quantization
    # -----------------------------------------------------------------
    # Median Filter (kernel_size=11 frames): Replaces transient 1-frame spikes
    # with the median of the surrounding window, preserving sharp note transitions.
    f0_smooth = medfilt(result.pitch_hz, kernel_size=11)

    # Convert smooth frequencies (Hz) into exact MIDI integers.
    # np.round() snaps micro-tonal vibrato to the nearest valid 12-tone musical pitch.
    midi_notes = librosa.hz_to_midi(f0_smooth)
    quantized_midi = np.round(midi_notes)

    # Debug: Print sample quantized frame
    print(
        f"Example MIDI frame: Time={result.timestamps[100]:.2f}s, MIDI={quantized_midi[100]}, Conf={result.confidence[100]:.2f}")

    # Optional visualization trigger
    # plot_pitch_data(result.timestamps, result.pitch_hz, f0_smooth, quantized_midi, result.confidence)

    # -----------------------------------------------------------------
    # 4. Note Segmentation (State Machine)
    # -----------------------------------------------------------------
    notes = []
    current_midi = None
    start_time = None

    # Iterate chronologically through every frame of the processed audio
    for t, midi_val, conf in zip(result.timestamps, quantized_midi, result.confidence):

        # Gate 1: Confidence > 0.5 ensures the model actually hears a note.
        # Gate 2: MIDI > 20 ignores impossible sub-bass frequencies (below ~25Hz).
        if conf > 0.5 and midi_val > 20:
            active_midi = midi_val
        else:
            active_midi = None

        # State transition detected: The note changed, or silence began/ended.
        if active_midi != current_midi:

            # If a note was actively being tracked, record its completion.
            if current_midi is not None:
                duration = t - start_time
                notes.append({
                    "note": librosa.midi_to_note(current_midi),
                    "start_time": round(start_time, 3),
                    "duration": round(duration, 3)
                })

            # Initialize tracking for the new state (a new MIDI value, or None for silence).
            current_midi = active_midi
            start_time = t

    # -----------------------------------------------------------------
    # 5. End-of-File Handling & Final Cleanup
    # -----------------------------------------------------------------
    # If the file ended while a note was still active, close and record it.
    if current_midi is not None:
        duration = result.timestamps[-1] - start_time
        notes.append({
            "note": librosa.midi_to_note(current_midi),
            "start_time": round(start_time, 3),
            "duration": round(duration, 3)
        })

    # Duration Filter: Purge notes shorter than 150ms.
    # This aggressively removes finger-clacks, fret buzz, and brief overtone hallucinations.
    cleaned_notes = [n for n in notes if n["duration"] >= 0.15]

    print(
        f"🎵 Detected {len(cleaned_notes)} notes after filtering short durations. Notes: {cleaned_notes}")

    return cleaned_notes
