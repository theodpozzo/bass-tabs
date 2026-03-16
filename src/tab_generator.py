"""
Module for converting musical notes into bass guitar tablature.
Applies fretboard optimization logic and formats the output into readable ASCII text 
with rhythmic spacing.
"""
import librosa

# Standard EADG Bass Tuning in MIDI values
# The bottom line represents the E string.
# The top line represents the G string.
TUNING_MIDI = {
    'G': 43,
    'D': 38,
    'A': 33,
    'E': 28
}

# Most bass guitars have around 20-24 frets, but mine has 20, so we'll use that as a hard limit.
MAX_FRET = 20


def map_notes_to_frets(notes: list[dict]) -> list[dict]:
    """
    Generalized fretboard mapping algorithm.
    Uses a dynamic greedy approach to minimize hand movement and string skipping 
    across the fretboard without any hardcoded anchor points.
    """
    fretted_notes = []

    # We don't know where the hand starts. We initialize to None.
    # The first note will establish our starting position organically.
    current_fret = None

    # Map strings to an index to calculate vertical string-skipping costs
    string_indices = {'G': 0, 'D': 1, 'A': 2, 'E': 3}
    current_string_idx = None

    for note_data in notes:
        target_midi = librosa.note_to_midi(note_data['note'])
        possible_positions = []

        for string_name, string_midi in TUNING_MIDI.items():
            fret_needed = target_midi - string_midi
            if 0 <= fret_needed <= MAX_FRET:
                possible_positions.append({
                    'string': string_name,
                    'fret': fret_needed,
                    'string_idx': string_indices[string_name]
                })

        # Fallback for sub-bass notes below the E string (Drop tuning)
        if not possible_positions:
            fallback = {'string': 'E', 'fret': int(
                target_midi - 28), 'string_idx': 3}
            note_data['string'] = fallback['string']
            note_data['fret'] = fallback['fret']
            fretted_notes.append(note_data)
            continue

        # --- DYNAMIC INITIALIZATION ---
        # If this is the very first note, pick the most "standard" starting position
        # We prefer fretted notes over open strings to establish a solid hand shape.
        if current_fret is None:
            best_start = min(possible_positions,
                             key=lambda p: p['fret'] if p['fret'] > 0 else 99)
            if best_start['fret'] == 99:  # Failsafe if ONLY open strings were available
                best_start = min(possible_positions, key=lambda p: p['fret'])

            current_fret = best_start['fret']
            current_string_idx = best_start['string_idx']

        # --- THE COST FUNCTION ---
        best_position = None
        lowest_cost = float('inf')

        for pos in possible_positions:
            fret = pos['fret']
            str_idx = pos['string_idx']
            cost = 0

            # 1. Horizontal Hand Movement
            # Bassists comfortably pivot within a 3-fret span.
            distance = abs(fret - current_fret)
            if distance > 2 and fret != 0:
                # Exponential penalty for big slides
                cost += (distance - 2) * 5

            # 2. Vertical String Skipping
            # Crossing from E to G string is harder than E to A.
            if current_string_idx is not None:
                str_distance = abs(str_idx - current_string_idx)
                cost += str_distance * 1.5

            # 3. Contextual Open Strings
            # Open strings are great if playing low on the neck (frets 1-4).
            # They are terrible if the hand is all the way up at the 12th fret.
            if fret == 0:
                if current_fret > 5:
                    cost += 10  # Heavy penalty: Don't jump from fret 12 to open string
                else:
                    cost += 1  # Slight penalty to keep tone consistent within boxes

            # 4. High Fret Penalty (Tone logic)
            # Bassists generally prefer thicker strings at lower frets over thin strings high up.
            if fret > 10:
                cost += (fret - 10) * 0.5

            if cost < lowest_cost:
                lowest_cost = cost
                best_position = pos

        # --- SAVE THE OPTIMIZED POSITION ---
        note_data['string'] = best_position['string']
        note_data['fret'] = best_position['fret']
        fretted_notes.append(note_data)

        # Update the hand's center of gravity dynamically
        # (We ignore open strings because they don't force the hand to move)
        if best_position['fret'] != 0:
            current_fret = best_position['fret']
        current_string_idx = best_position['string_idx']

    return fretted_notes


def generate_tab_text(fretted_notes: list[dict], song_name: str) -> str:
    """
    Generate perfectly aligned ASCII bass tablature.

    Features
    --------
    • Fixed-width time grid
    • Multi-digit fret support
    • 4 bars per line
    • Perfect alignment across strings
    • Bar separators
    """

    if not fretted_notes:
        return "No notes detected."

    header = (
        "Bass tabs generated by Theo's AI Bass Tab Generator\n"
        f"Song Name: {song_name}\n\n"
    )

    # -------------------------
    # Timing parameters
    # -------------------------

    TIME_PER_STEP = 0.05        # resolution of grid (seconds)
    BAR_LENGTH = 0.5            # 4/4 bar at 120 bpm
    BARS_PER_LINE = 4

    STEPS_PER_BAR = int(BAR_LENGTH / TIME_PER_STEP)
    STEPS_PER_LINE = STEPS_PER_BAR * BARS_PER_LINE

    # Determine song duration
    last = fretted_notes[-1]
    total_duration = last["start_time"] + last["duration"]

    total_steps = int(total_duration / TIME_PER_STEP) + 1

    # -------------------------
    # Build empty grid
    # -------------------------

    grid = {
        "G": ["--"] * total_steps,
        "D": ["--"] * total_steps,
        "A": ["--"] * total_steps,
        "E": ["--"] * total_steps,
    }

    # -------------------------
    # Place notes into grid
    # -------------------------

    for note in fretted_notes:

        step = int(note["start_time"] / TIME_PER_STEP)

        string = note["string"]
        fret = str(note["fret"])

        # Normalize width (always 2 chars)
        fret_cell = fret.rjust(2, "-")

        if step < total_steps:
            grid[string][step] = fret_cell

    # -------------------------
    # Render lines
    # -------------------------

    output = header

    num_lines = (total_steps // STEPS_PER_LINE) + 1

    for line_index in range(num_lines):

        start = line_index * STEPS_PER_LINE
        end = start + STEPS_PER_LINE

        for string in ["G", "D", "A", "E"]:

            line = f"{string}|"

            for step in range(start, min(end, total_steps)):

                # Insert bar separators
                if step != start and (step - start) % STEPS_PER_BAR == 0:
                    line += "|"

                line += grid[string][step]

            line += "|"
            output += line + "\n"

        output += "\n"

    return output
