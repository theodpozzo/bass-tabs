# 🎸 Bass Tab Generator

An automated pipeline that takes a mixed audio track (like an MP3 or WAV), isolates the bass guitar, analyzes the pitches, and generates a text-based bass tablature. 

Born out of the frustration of not being able to find a proper bass tab for *Strangers To Neighbours* by The Vanns, this project combines Machine Learning (Music Information Retrieval) with algorithmic fretboard logic to transcribe basslines automatically.

## ✨ Features
* **Source Separation:** Uses Facebook's `demucs` to isolate the bass track from a fully mixed audio file.
* **Accurate Pitch Detection:** Leverages `torchcrepe` (a PyTorch-based neural network) to analyze the isolated bass stem and extract precise musical notes.
* **Algorithmic Fretboard Mapping:** Translates a sequence of musical pitches into logical string and fret combinations for standard EADG bass tuning.
* **CLI Interface:** Easy-to-use command-line interface for batch processing songs.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Audio Processing:** `librosa`, `soundfile`, FFmpeg
* **Machine Learning:** PyTorch, `demucs`, `torchcrepe`
* **Data Handling:** `numpy`, `scipy`

## 📂 Project Structure
```text
bass-tabs/
├── data/
│   ├── inputs/          # Place your raw MP3/WAV files here
│   ├── stems/           # Isolated bass stems are saved here
│   └── outputs/         # Generated text tabs are saved here
├── src/
│   ├── audio_processor.py   # Handles demucs source separation
│   ├── pitch_detector.py    # Handles torchcrepe pitch extraction
│   ├── tab_generator.py     # Handles string/fret logic & formatting
│   └── main.py              # The CLI orchestrator
├── requirements.txt         # Python dependencies
├── .gitignore               
└── README.md                

```

## 🚀 Getting Started

### 1. System Requirements

This project relies on `demucs` and `librosa`, which require **FFmpeg** to handle audio files under the hood.

* **Linux (Ubuntu/Mint):** `sudo apt update && sudo apt install ffmpeg`
* **Mac:** `brew install ffmpeg`
* **Windows:** Download from the [FFmpeg website](https://ffmpeg.org/download.html) and add it to your system PATH.

### 2. Python Setup

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/yourusername/bass-tabs.git](https://github.com/yourusername/bass-tabs.git)
cd bass-tabs

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

```

*(Note: If you have an NVIDIA GPU, PyTorch and Demucs will automatically leverage it for significantly faster processing times!)*

## 🎧 Usage

To generate a bass tab, run the orchestrator script and pass the path to your audio file:

```bash
python src/main.py data/inputs/strangers_to_neighbours.mp3

```

By default, the text tab will be saved to `data/outputs/`. You can specify a custom output directory using the `--output` flag:

```bash
python src/main.py data/inputs/song.wav --output custom/path/

```

## 🗺️ Roadmap & Future Expansions
[x]
* [ ] Audio source separation pipeline
* [ ] Deep learning pitch detection
* [ ] Basic EADG fretboard mapping
* [ ] Add support for 5-string and Drop D tunings
* [ ] **Generative ML:** Train a model on the output tabs to generate completely original, cool basslines based on inputted chord progressions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
