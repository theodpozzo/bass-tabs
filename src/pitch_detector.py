"""
Pitch extraction module for bass audio.

This module analyses an audio waveform and converts it into a sequence
of musical notes with start times and durations.

Pipeline
--------
Audio file
    ↓
Pitch detection (librosa YIN)
    ↓
Median filtering (smooth pitch jitter)
    ↓
Frequency → note conversion
    ↓
Note segmentation
    ↓
Short-note filtering
"""

import librosa
import numpy as np
from scipy.signal import medfilt


# ---------------------------------------------------------------------
# Utility: Convert frequency (Hz) to a musical note name
# ---------------------------------------------------------------------
def freq_to_note(freq: float) -> str | None:
    """
    Converts a frequency value into a musical note.

    Parameters
    ----------
    freq : float
        Frequency in Hz.

    Returns
    -------
    str | None
        Musical note name (e.g. 'E2') or None if frequency is invalid.
    """

    # YIN sometimes returns NaN when pitch is not detected
    if np.isnan(freq) or freq <= 0:
        return None

    return librosa.hz_to_note(freq)


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

    print(f"Loading audio for pitch detection: {audio_path}")

    # -----------------------------------------------------------------
    # Load audio
    # -----------------------------------------------------------------
    # 16 kHz is sufficient for bass analysis and keeps computation low
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Hop length determines time resolution of pitch frames
    hop_length = 256

    # -----------------------------------------------------------------
    # Pitch detection using the YIN algorithm
    # -----------------------------------------------------------------
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("E1"),   # lowest bass note
        fmax=librosa.note_to_hz("C4"),   # safe upper bound
        sr=sr,
        hop_length=hop_length
    )

    # Convert frame indices to timestamps
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # -----------------------------------------------------------------
    # Smooth pitch curve
    # -----------------------------------------------------------------
    # Median filtering removes rapid jitter between neighbouring notes
    f0_smooth = medfilt(f0, kernel_size=9)

    # -----------------------------------------------------------------
    # Convert frequency sequence → note segments
    # -----------------------------------------------------------------
    notes = []

    current_note = None
    start_time = None

    for t, freq in zip(times, f0_smooth):

        note = freq_to_note(freq)

        # When note changes, close the previous note segment
        if note != current_note:

            if current_note is not None:

                duration = t - start_time

                notes.append({
                    "note": current_note,
                    "start_time": start_time,
                    "duration": duration
                })

            # Start a new note
            current_note = note
            start_time = t

    # Handle final note
    if current_note is not None:
        duration = times[-1] - start_time

        notes.append({
            "note": current_note,
            "start_time": start_time,
            "duration": duration
        })

    # -----------------------------------------------------------------
    # Remove extremely short notes (pitch detection artefacts)
    # -----------------------------------------------------------------
    MIN_NOTE_DURATION = 0.08   # seconds

    cleaned_notes = [
        n for n in notes if n["duration"] >= MIN_NOTE_DURATION
    ]

    # -----------------------------------------------------------------
    # Debug print
    # -----------------------------------------------------------------
    for n in cleaned_notes:
        start = n["start_time"]
        duration = n["duration"]
        end = start + duration

        print(f"{start:.2f}–{end:.2f} : {n['note']} ({duration:.2f}s)")

    return cleaned_notes
