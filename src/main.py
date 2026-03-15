"""
Main entry point for the Bass Tab Generator.
Orchestrates the pipeline: Audio Input -> Separation -> Pitch Detection -> Tab Output.
"""

import argparse
from pathlib import Path

# Import the functions you will be building in the other files!
from audio_processor import separate_bass
from pitch_detector import extract_notes
from tab_generator import map_notes_to_frets, generate_tab_text


def process_song(input_path: str, output_dir: str) -> None:
    """
    Runs the full pipeline to convert an audio file into a bass tab.
    """
    input_file = Path(input_path)
    out_dir = Path(output_dir)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎸 Starting processing for: {input_file.name}")

    # --- STEP 1: Audio Separation ---
    print("[1/4] Isolating bass track with Demucs...")
    stems_dir = Path("data/stems")
    stems_dir.mkdir(parents=True, exist_ok=True)

    # bass_stem_path = separate_bass(str(input_file), str(stems_dir))

    # --- STEP 2: Pitch Detection ---
    print("[2/4] Analyzing pitch using Torchcrepe...")
    # raw_notes = extract_notes(bass_stem_path)
    raw_notes = extract_notes("data/stems/mdx_extra/Come Together/bass.mp3")

    if not raw_notes:
        print("❌ No notes detected! Exiting.")
        return

    print(raw_notes)

    # --- STEP 3: Fretboard Logic ---
    print("[3/4] Mapping notes to bass fretboard...")
    fretted_notes = map_notes_to_frets(raw_notes)

    # --- STEP 4: Tab Generation ---
    print("[4/4] Generating text tablature...")
    tab_text = generate_tab_text(fretted_notes)

    # --- Save the Output ---
    output_file = out_dir / f"{input_file.stem}_bass_tab.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tab_text)

    print(f"✅ Success! Bass tab saved to: {output_file}")


if __name__ == "__main__":
    # Set up the Command Line Interface
    parser = argparse.ArgumentParser(
        description="Convert an audio file into a bass tab.")

    parser.add_argument(
        "input_audio",
        type=str,
        help="Path to the input audio file (e.g., data/inputs/strangers_to_neighbours.mp3)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs",
        help="Directory to save the generated tab (default: data/outputs)"
    )

    args = parser.parse_args()

    # Run the orchestrator
    process_song(args.input_audio, args.output)
