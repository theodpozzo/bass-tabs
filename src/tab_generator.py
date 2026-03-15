"""
Module for converting musical notes into bass guitar tablature.
Applies fretboard logic and formats the output into readable text.
"""

import librosa


# ------------------------------------------------------------------
# Standard bass tuning (MIDI note numbers)
# ------------------------------------------------------------------

BASS_STRINGS = {
    "E": librosa.note_to_midi("E1"),
    "A": librosa.note_to_midi("A1"),
    "D": librosa.note_to_midi("D2"),
    "G": librosa.note_to_midi("G2"),
}

MAX_FRET = 20


# ------------------------------------------------------------------
# Determine playable fretboard positions
# ------------------------------------------------------------------

def map_notes_to_frets(notes: list[dict]) -> list[dict]:
    """
    Converts detected notes into playable bass string/fret positions.

    Strategy
    --------
    For each note:
    1. Convert note name → MIDI pitch
    2. Check each bass string to see if the note is reachable
    3. Choose the position with the lowest fret number

    Parameters
    ----------
    notes : list[dict]

        Example input element:
        {
            "note": "F2",
            "start_time": 1.2,
            "duration": 0.4
        }

    Returns
    -------
    list[dict]

        Example output element:
        {
            "note": "F2",
            "start_time": 1.2,
            "duration": 0.4,
            "string": "E",
            "fret": 1
        }
    """

    fretted_notes = []

    for n in notes:

        note_name = n["note"]
        midi_note = librosa.note_to_midi(note_name)

        candidates = []

        # Check each string to see if the note can be played
        for string, open_midi in BASS_STRINGS.items():

            fret = midi_note - open_midi

            if 0 <= fret <= MAX_FRET:
                candidates.append((string, fret))

        # Skip if note is outside bass range
        if not candidates:
            continue

        # Choose the lowest fret position
        string, fret = min(candidates, key=lambda x: x[1])

        fretted_note = {
            **n,
            "string": string,
            "fret": fret
        }

        fretted_notes.append(fretted_note)

    return fretted_notes


# ------------------------------------------------------------------
# Convert fretted notes into ASCII bass tab
# ------------------------------------------------------------------

def generate_tab_text(fretted_notes: list[dict]) -> str:
    """
    Converts fretted notes into a simple ASCII bass tab.

    Notes are placed sequentially across the tab.
    Timing is approximate — each note occupies a fixed slot.

    Parameters
    ----------
    fretted_notes : list[dict]

    Returns
    -------
    str
        Multiline string representing the bass tab.
    """

    # Create empty tab lines
    tab = {
        "G": [],
        "D": [],
        "A": [],
        "E": []
    }

    for note in fretted_notes:

        string = note["string"]
        fret = str(note["fret"])

        # Width ensures alignment for double-digit frets
        width = max(2, len(fret))

        for s in tab:

            if s == string:
                tab[s].append(fret.ljust(width, "-"))
            else:
                tab[s].append("-" * width)

    # Join each string into a line
    lines = []

    for s in ["G", "D", "A", "E"]:
        line = s + "|" + "".join(tab[s])
        lines.append(line)

    return "\n".join(lines)
