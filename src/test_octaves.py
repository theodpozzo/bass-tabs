import numpy as np
import soundfile as sf
import librosa
from pitch_detector import extract_notes


def generate_overtone_trap_test(filename="octave_test.wav"):
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # E1 Fundamental (41.2 Hz) - We'll make this intentionally quiet (amplitude 0.3)
    fundamental = 0.3 * np.sin(2 * np.pi * 41.20 * t)

    # E2 Overtone (82.4 Hz) - We'll make this intentionally LOUD (amplitude 0.8)
    # This simulates a real bass string where the 2nd harmonic dominates
    overtone = 0.8 * np.sin(2 * np.pi * 82.41 * t)

    # Combine them
    mixed_signal = fundamental + overtone

    sf.write(filename, mixed_signal, sr)
    print(f"Generated {filename}. Run this through extract_notes()!")
    extracted_notes = extract_notes(filename)
    print("Extracted Notes:", extracted_notes)


if __name__ == "__main__":
    generate_overtone_trap_test()
