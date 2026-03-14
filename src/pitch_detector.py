"""
Module for analyzing isolated bass audio and extracting pitch data.
Converts audio waveforms into a sequence of musical notes.
"""


def extract_notes(audio_path: str) -> list[dict]:
    """
    Analyzes an audio file to determine the sequence of pitches played.

    Args:
        audio_path (str): Path to the isolated bass audio file.

    Returns:
        list[dict]: A list of dictionaries, where each dict represents a note
                    e.g., {'note': 'E2', 'start_time': 1.2, 'duration': 0.5}
    """
    pass
