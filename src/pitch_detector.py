"""
Module for analyzing isolated bass audio and extracting pitch data.
Converts audio waveforms into a sequence of musical notes.
"""
import librosa
import torch
import torchcrepe
from scipy.signal import medfilt
import numpy as np


# 3. Convert frequency → musical note
def freq_to_note(freq):
    if np.isnan(freq):
        return None
    return librosa.hz_to_note(freq)


def extract_notes(audio_path: str) -> list[dict]:
    """
    Analyzes an audio file to determine the sequence of pitches played.

    Args:
        audio_path (str): Path to the isolated bass audio file.

    Returns:
        list[dict]: A list of dictionaries, where each dict represents a note
                    e.g., {'note': 'E2', 'start_time': 1.2, 'duration': 0.5}
    """

    # 2. Load the audio and detect pitch
    print(f"Loading audio for pitch detection: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)

    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("E1"),
        fmax=librosa.note_to_hz("C4"),
        sr=sr
    )

    times = librosa.times_like(f0, sr=sr)

    # 4. Smooth the pitch (very important)
    f0_smooth = medfilt(f0, kernel_size=9)

    # 5. Convert frames into note segments
    notes = []
    current_note = None
    start_time = None

    for t, freq in zip(times, f0_smooth):

        note = freq_to_note(freq)

        if note != current_note:
            if current_note is not None:
                notes.append((start_time, t, current_note))

            current_note = note
            start_time = t

    if current_note is not None:
        notes.append((start_time, times[-1], current_note))

    # Print the detected notes for debugging
    for start, end, note in notes:
        duration = end - start
        print(f"{start:.2f}–{end:.2f} : {note} ({duration:.2f}s)")

    return notes
