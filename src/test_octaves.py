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
    print("Expected Output in the same format as extract_notes():")
    print("[{'note': 'E1', 'start_time': 0.0, 'duration': 2.0}, {'note': 'E2', 'start_time': 0.0, 'duration': 2.0}]")
    print("Note: The test is designed to see if the model can detect the fundamental E1 even when the E2 overtone is louder. A perfect model would ideally detect both, but we want to ensure it doesn't just pick the overtone and miss the fundamental.")
    print("In practice, due to the nature of pitch detection and the dominance of the overtone, some models might only detect E2. This test will help us evaluate how well the model handles such scenarios.")
    print("If the output only contains E2, it indicates that the model is being misled by the overtone and missing the fundamental, which is a common issue in pitch detection. If it detects both E1 and E2, that would be ideal. If it only detects E1, that would be surprising and might indicate an issue with the overtone generation or the model's sensitivity to higher harmonics.")
    print("In summary, this test is a way to evaluate the robustness of the pitch detection model in scenarios where overtones are present, which is a common occurrence in real bass recordings.")
    print("Expected Output (if the model is misled by the overtone):")
    print("""[
    {'note': 'E2', 'start_time': 0.0, 'duration': 2.0}
    ]""")
    print("Expected Output (if the model detects both):")
    print("""[
    {'note': 'E1', 'start_time': 0.0, 'duration': 2.0},
    {'note': 'E2', 'start_time': 0.0, 'duration': 2.0}
    ]""")
    extracted_notes = extract_notes(filename)
    print("Extracted Notes:", extracted_notes)


if __name__ == "__main__":
    generate_overtone_trap_test()
