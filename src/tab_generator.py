"""
Module for converting musical notes into bass guitar tablature.
Applies fretboard logic and formats the output into readable text.
"""


def map_notes_to_frets(notes: list[dict]) -> list[dict]:
    """
    Takes a sequence of notes and determines the optimal string and fret 
    for each, based on standard bass tuning (E A D G).

    Args:
        notes (list[dict]): The sequence of notes from the pitch detector.

    Returns:
        list[dict]: The same list, but with 'string' and 'fret' keys added.
    """
    pass


def generate_tab_text(fretted_notes: list[dict]) -> str:
    """
    Formats a sequence of fretted notes into a standard text-based bass tab.

    Args:
        fretted_notes (list[dict]): Notes with string/fret data included.

    Returns:
        str: The formatted multiline string representing the bass tab.
    """
    pass
