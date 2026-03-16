<markdown>

# 🎸 AI Bass Tab Generator

An automated machine learning pipeline that transcribes raw audio files into highly accurate, rhythmically spaced, and ergonomically mapped bass guitar tablature.

Built to solve the "Missing Tab" problem for niche songs, this tool isolates the bass track from any audio file, extracts the fundamental pitches using state-of-the-art neural networks, and uses a dynamic cost-function algorithm to map the notes to the most natural physical fretboard positions.

## ✨ Features

* **Audio Source Separation:** Uses Meta's `demucs` to cleanly isolate the bass stem from full-band mixes (handling drum bleed and competing frequencies).
* **State-of-the-Art Pitch Detection:** Upgraded to **SwiftF0**, a lightweight (95k parameter) ONNX model that is ~42x faster than CREPE and highly resistant to the heavy overtone hallucinations common in bass guitar frequencies.
* **Ergonomic Fretboard Mapping:** Replaces naive string-jumping with a **Dynamic Greedy Cost Function**. The algorithm evaluates horizontal hand movement, vertical string-skipping, and contextual open-string penalties to generate tight, playable "box shapes."
* **Rhythmic Formatting:** Unlike naive tab generators, the text output includes a rigid chronological grid (where 1 dash `-` = 0.1 seconds) and automatic measure lines, allowing the player to actually read the rhythm of the performance.
* **Batch Processing:** Seamlessly process a single `.mp3` or an entire directory of tracks with one command.

## 🧠 Pipeline Architecture

```text
Audio File (mp3/wav)
    │
    ▼
1. Source Separation (Demucs) ──▶ Isolates raw bass stem
    │
    ▼
2. Pitch Detection (SwiftF0) ──▶ Neural network extracts raw Hz and confidence scores
    │
    ▼
3. Signal Processing ──▶ Amplitude gating, median filtering, and transient noise suppression
    │
    ▼
4. MIDI Quantization ──▶ Snaps micro-tonal vibrato to the 12-tone musical grid
    │
    ▼
5. Fretboard Optimization ──▶ Cost-function maps notes to optimal E-A-D-G string combinations
    │
    ▼
Text Output (.txt) ──▶ Rhythmically spaced ASCII Tablature

```

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/theodpozzo/bass-tabs.git
cd bass-tabs

```


2. **Set up a virtual environment (Recommended):**
```bash
python3 -m venv .venv
source .venv/bin/activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


*Required packages include: `librosa`, `numpy`, `scipy`, `matplotlib`, `demucs`, and `swiftf0`.*

## 🚀 Usage

The Command Line Interface (CLI) is dynamically routed. You can pass it a single audio file or a folder containing multiple tracks.

**Transcribe a single song:**

```bash
python src/main.py "data/inputs/Strangers To Neighbours.mp3"

```

**Batch process an entire directory:**

```bash
python src/main.py data/inputs/ --output data/outputs/

```

### Example Output

The generated tab (`SongName_bass_tab.txt`) will look like this, featuring accurate box shapes and proportional rhythm spacing:

```text
Song Name: Feel Good Inc (intro)

G|--------------------|--------------------|--------------------|-------------------3|
D|---1----------------|---------1---------3|-------4------------|--------------------|
A|--------------------|--------------------|--------------------|---2----------------|
E|--------------------|--------------------|--------------------|--------------------|

```

## 🛠️ Advanced Details

### The "Octave Error" Solution

Bass strings generate massive overtones (e.g., the 2nd harmonic is often louder than the fundamental root). This pipeline tackles this via:

1. **SwiftF0:** Known for an Octave Accuracy (OA) of 96.75% on clean audio.
2. **Algorithmic Gravity:** A heavy cost penalty in `tab_generator.py` suppresses sudden, unnatural 12-fret jumps, effectively clamping down on any residual overtone hallucinations.

### Visualization & Debugging

If you need to tune the duration filters or visualize the pitch tracking, `src/pitch_detector.py` includes an optional Matplotlib function (`plot_pitch_data`) to graph the raw Hz, the median-smoothed Hz, the quantized MIDI stair-steps, and the SwiftF0 gating confidence.

### 🔮 Future Roadmap: True Machine Learning Tablature Inference

While the current Dynamic Greedy Cost Function effectively calculates physical fretboard distances, the next major evolution for this project is to replace the heuristic physics engine with a fully supervised Deep Learning model (such as a Transformer) to handle tablature generation.

1. **The Training Data:** I plan to build a custom dataset mapping thousands of human-transcribed bass tabs to their corresponding raw MIDI sequences (using [MIDI] as the input X and [String, Fret] combinations as the target Y).

2. **Context-Aware AI:** Instead of relying on hardcoded math to find the shortest distance, a sequence-modeling neural network will learn the actual "grammar" and stylistic nuances of human bass players. By analyzing the contextual sequence of notes around a target pitch, the model will organically understand when to stay in a tight box shape, and when a bassist would naturally choose to slide up a thicker string for a warmer tone.

3. **The End Goal:** A seamless, end-to-end AI pipeline where raw audio is converted to MIDI via SwiftF0, and a trained Machine Learning model instantly predicts the most natural, human-feeling tablature based on real-world ergonomic data.