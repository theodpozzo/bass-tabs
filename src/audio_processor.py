"""
Module for handling audio file manipulation and source separation.
Isolates the bass track from a full mixed song.
"""
import demucs.separate
from pathlib import Path


def separate_bass(input_path: str, output_dir: str) -> str:
    """
    Uses source separation to extract the bass stem from an audio file.

    Args:
        input_path (str): Path to the original mixed audio file.
        output_dir (str): Directory to save the isolated stem.

    Returns:
        str: The file path to the newly isolated bass audio file.
    """
    # Build the Demucs arguments
    args = [
        "--mp3",               # Output as MP3 to save space
        "--two-stems", "bass",  # Only extract the bass, ignore drums/vocals/other
        "-n", "mdx_extra",     # The specific Demucs ML model to use
        "-d", "cpu",           # FIX: Force CPU to bypass the GTX 960 CUDA crash
        "-o", output_dir,      # Route the output directly to our data/stems folder
        input_path
    ]

    # Run Demucs source separation
    demucs.separate.main(args)

    # Calculate the exact path where Demucs saved the file
    # Structure: output_dir / model_name / track_name / "bass.mp3"
    track_name = Path(input_path).stem
    expected_out_path = Path(output_dir) / "mdx_extra" / \
        track_name / "bass.mp3"

    return str(expected_out_path)
