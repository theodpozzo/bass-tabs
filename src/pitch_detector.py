"""
Module for analyzing isolated bass audio and extracting pitch data.
Converts audio waveforms into a sequence of musical notes.
"""
import librosa
import torch
import torchcrepe


def extract_notes(audio_path: str) -> list[dict]:
    """
    Analyzes an audio file to determine the sequence of pitches played.

    Args:
        audio_path (str): Path to the isolated bass audio file.

    Returns:
        list[dict]: A list of dictionaries, where each dict represents a note
                    e.g., {'note': 'E2', 'start_time': 1.2, 'duration': 0.5}
    """
    print(f"Loading audio for pitch detection: {audio_path}")

    # 1. Load the audio directly using librosa at 16kHz (crepe's native rate)
    audio, sr = librosa.load(audio_path, sr=16000)

    # torchcrepe requires the audio to be a PyTorch tensor of shape (batch, samples)
    audio_tensor = torch.tensor(audio).unsqueeze(0)

    # 2. Run torchcrepe frame-by-frame pitch tracking
    print("Running neural network pitch detection via torchcrepe...")
    time, frequency, confidence, activation = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=int(sr / 100),  # 10ms steps (100 frames per second)
        fmin=20.0,                # 20Hz (Sub-bass range)
        fmax=400.0,               # 400Hz (Upper bass range)
        model='full',             # Use the most accurate model
        device='cpu',             # FIX: Force CPU to bypass GTX 960 CUDA crash
        batch_size=512           # FIX: Process in small chunks to save RAM!
    )
    print("Pitch detection complete. Processing results...")

    # Convert the PyTorch tensors back into flat 1D Python/Numpy lists
    frequencies = frequencies.squeeze().numpy()
    confidences = confidences.squeeze().numpy()

    # Generate a time array based on the 10ms frame steps
    time = [i * 0.01 for i in range(len(frequencies))]

    notes_list = []
    current_note = None
    start_time = 0.0

    # 3. Iterate through every single frame
    for t, freq, conf in zip(time, frequencies, confidences):

        # Filter out background noise/silence using the confidence score
        if conf > 0.5:
            note_name = librosa.hz_to_note(freq)
        else:
            note_name = None  # Silence

        # 4. Grouping Logic: Did the note change (or start/stop)?
        if note_name != current_note:

            # If we were tracking a note, it just ended. Let's save it!
            if current_note is not None:
                duration = t - start_time

                # Filter out micro-glitches (ignore notes shorter than 50ms)
                if duration > 0.05:
                    notes_list.append({
                        'note': current_note,
                        'start_time': round(start_time, 3),
                        'duration': round(duration, 3)
                    })

            print(
                f"Detected note change at {t:.2f}s: {current_note} -> {note_name}")

            # Start tracking the new note (or silence)
            current_note = note_name
            start_time = t

    # Catch the very last note if the song ends while the bass is still ringing out
    if current_note is not None:
        duration = time[-1] - start_time
        if duration > 0.05:
            notes_list.append({
                'note': current_note,
                'start_time': round(start_time, 3),
                'duration': round(duration, 3)
            })

    return notes_list
