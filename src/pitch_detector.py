"""
Pitch extraction module for bass audio.

This module analyses an audio waveform and converts it into a sequence
of musical notes with start times and durations.

Pipeline
--------
Audio file
    ↓
Pitch detection (SwiftF0 - lightweight deep learning)
    ↓
Confidence filtering (removes silence/noise)
    ↓
Median filtering (smooth pitch jitter)
    ↓
Frequency → MIDI Quantization (snaps vibrato to grid)
    ↓
Note segmentation
    ↓
Short-note filtering
"""

import librosa
import numpy as np
from scipy.signal import medfilt
from swift_f0 import *  # Modern, lightweight replacement for CREPE

# ---------------------------------------------------------------------
# Main pitch extraction function
# ---------------------------------------------------------------------


def extract_notes(audio_path: str) -> list[dict]:
    """
    Detects notes played in an isolated bass recording.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.

    Returns
    -------
    list[dict]
        Each dictionary represents a detected note:

        {
            "note": "E2",
            "start_time": 1.20,
            "duration": 0.45
        }
    """
    # -----------------------------------------------------------------
    # Load audio
    # -----------------------------------------------------------------
    # 16 kHz is typically standard for pitch models; SwiftF0 handles it well
    audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)

    # -----------------------------------------------------------------
    # Pitch detection using SwiftF0 (Deep Learning)
    # -----------------------------------------------------------------

    # Initialize the SwiftF0 model.
    # It is significantly lighter (only ~95k parameters) and faster than CREPE.
    pitch_estimator = SwiftF0(fmin=SwiftF0.DEFAULT_FMIN, fmax=SwiftF0.DEFAULT_FMAX,
                              confidence_threshold=SwiftF0.DEFAULT_CONFIDENCE_THRESHOLD)

    result = pitch_estimator.detect_from_array(audio_data, sample_rate)

    # SwiftF0 natively takes the 1D numpy array (no PyTorch tensor conversion needed).
    # It usually returns time, frequency, and confidence arrays directly.
    # Note: Check your specific SwiftF0 version docs if the return order varies slightly.

    # -----------------------------------------------------------------
    # Smooth pitch curve
    # -----------------------------------------------------------------
    # Apply a 110ms median filter to remove sudden 1-frame glitches
    f0_smooth = medfilt(result.pitch_hz, kernel_size=11)

    # -----------------------------------------------------------------
    # Convert frequency sequence → quantized MIDI grid
    # -----------------------------------------------------------------
    # This prevents vibrato from splitting one note into multiple notes
    midi_notes = librosa.hz_to_midi(f0_smooth)
    quantized_midi = np.round(midi_notes)

    # -----------------------------------------------------------------
    # Note segmentation
    # -----------------------------------------------------------------
    notes = []
    current_midi = None
    start_time = None

    for t, midi_val, conf in zip(result.timestamps, quantized_midi, result.confidence):

        # SwiftF0 outputs confidence scores we can use to filter silence.
        # We also ignore extreme sub-bass artifacts (< MIDI 20)
        if conf > 0.5 and midi_val > 20:
            active_midi = midi_val
        else:
            active_midi = None

        # When note changes, close the previous note segment
        if active_midi != current_midi:
            if current_midi is not None:
                duration = t - start_time

                notes.append({
                    "note": librosa.midi_to_note(current_midi),
                    "start_time": round(start_time, 3),
                    "duration": round(duration, 3)
                })

            # Start a new note
            current_midi = active_midi
            start_time = t

    # Handle the last note if it was still active at the end of the audio
    if current_midi is not None:
        duration = result.timestamps[-1] - start_time

        notes.append({
            "note": librosa.midi_to_note(current_midi),
            "start_time": round(start_time, 3),
            "duration": round(duration, 3)
        })
    # -----------------------------------------------------------------
    # Filter out very short notes (e.g., < 100ms) which are likely
    # spurious detections or noise.
    # -----------------------------------------------------------------
    cleaned_notes = [n for n in notes if n["duration"] >= 0.1]

    return cleaned_notes
