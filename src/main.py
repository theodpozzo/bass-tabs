"""
Main entry point for the Bass Tab Generator.
Orchestrates the pipeline: Audio Input -> Separation -> Pitch Detection -> Tab Output.
Supports processing individual audio files or batch-processing entire directories.
"""

import argparse
from pathlib import Path

# Import the core pipeline functions
from audio_processor import separate_bass
from pitch_detector import extract_notes
from tab_generator import map_notes_to_frets, generate_tab_text


def process_song(input_path: str | Path, output_dir: str | Path) -> None:
    """
    Runs the full pipeline to convert a single audio file into a bass tab.

    Args:
        input_path (str | Path): The path to the specific input audio file.
        output_dir (str | Path): The destination directory for the generated text tab.
    """
    input_file = Path(input_path)
    out_dir = Path(output_dir)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎸 Starting processing for: {input_file.name}")

    # --- STEP 1: Audio Separation ---
    print("[1/4] Isolating bass track with Demucs...")
    stems_dir = Path("data/stems")
    stems_dir.mkdir(parents=True, exist_ok=True)

    bass_stem_path = separate_bass(str(input_file), str(stems_dir))

    # --- STEP 2: Pitch Detection ---
    print("[2/4] Analyzing pitch using SwiftF0...")
    # raw_notes = extract_notes(str(input_file))
    raw_notes = extract_notes(bass_stem_path)

    if not raw_notes:
        print(f"❌ No notes detected in {input_file.name}! Skipping.")
        return

    # --- STEP 3: Fretboard Logic ---
    print("[3/4] Mapping notes to bass fretboard...")
    fretted_notes = map_notes_to_frets(raw_notes)

    # --- STEP 4: Tab Generation ---
    print("[4/4] Generating text tablature...")
    tab_text = generate_tab_text(fretted_notes, input_file.stem)

    # --- Save the Output ---
    output_file = out_dir / f"{input_file.stem}_bass_tab.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(tab_text)

    print(f"✅ Success! Bass tab saved to: {output_file}")


def process_directory(input_dir: str | Path, output_dir: str | Path) -> None:
    """
    Scans a directory for supported audio files and batch-processes each one.

    Args:
        input_dir (str | Path): The directory containing the audio files.
        output_dir (str | Path): The destination directory for the generated text tabs.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    # Supported audio formats
    valid_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

    print(f"\n📂 Scanning directory: {in_dir}")

    # Find all matching files
    audio_files = [f for f in in_dir.iterdir() if f.is_file()
                   and f.suffix.lower() in valid_extensions]

    if not audio_files:
        print(f"⚠️ No supported audio files found in {in_dir}.")
        return

    print(f"Found {len(audio_files)} tracks to process. Starting batch job...")

    # Process each file sequentially
    for audio_file in audio_files:
        process_song(audio_file, out_dir)

    print("\n🎉 Batch processing complete!")


if __name__ == "__main__":
    # Set up the Command Line Interface
    parser = argparse.ArgumentParser(
        description="Convert an audio file (or directory of audio files) into a bass tab."
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single audio file OR a directory containing audio files."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs",
        help="Directory to save the generated tab(s) (default: data/outputs)"
    )

    args = parser.parse_args()

    # Dynamic Path Routing
    target_path = Path(args.input_path)

    if target_path.is_file():
        # User passed a specific song
        process_song(target_path, args.output)

    elif target_path.is_dir():
        # User passed a folder
        process_directory(target_path, args.output)

    else:
        # Invalid path caught
        print(
            f"❌ Error: The path '{target_path}' does not exist or is inaccessible.")
