"""
Experimental ML Tablature Generator (Custom 20-Fret Bass Architecture).
Uses a Transformer Encoder (Masked Language Modeling approach) to infer 
ergonomic bass guitar string/fret combinations from raw pitch sequences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import math

# Automatically detect if a GPU is available (Colab) or fallback to CPU (Local)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Physical Bass Constraints & Vocabulary ---

TUNING_MIDI = {'E': 28, 'A': 33, 'D': 38, 'G': 43}
STRINGS = list(TUNING_MIDI.keys())
NUM_STRINGS = len(STRINGS)
POSITIONS_PER_STRING = 21

# 84 physical positions + Padding (84) + Unknown/Rest (85)
NUM_BASS_POSITIONS = (NUM_STRINGS * POSITIONS_PER_STRING) + 2
NUM_MIDI_NOTES = 128


class PositionalEncoding(nn.Module):
    """Injects sequence order into the model."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x


class BassTabTransformer(nn.Module):
    """The core BERT-style Transformer architecture."""

    def __init__(self, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.midi_embedding = nn.Embedding(NUM_MIDI_NOTES, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers)
        self.classifier = nn.Linear(d_model, NUM_BASS_POSITIONS)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.midi_embedding(
            src) * math.sqrt(self.midi_embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        logits = self.classifier(output)
        return logits


class BassTabDataset(Dataset):
    """A PyTorch Dataset that feeds MIDI sequences and target frets into the model."""

    def __init__(self, midi_sequences, target_positions, max_length=100):
        self.midi_sequences = midi_sequences
        self.target_positions = target_positions
        self.max_length = max_length
        self.pad_token = 84

    def __len__(self):
        return len(self.midi_sequences)

    def __getitem__(self, idx):
        midi = self.midi_sequences[idx]
        target = self.target_positions[idx]

        if len(midi) > self.max_length:
            midi = midi[:self.max_length]
            target = target[:self.max_length]

        padding_length = self.max_length - len(midi)
        midi = midi + [self.pad_token] * padding_length
        target = target + [self.pad_token] * padding_length

        return torch.tensor(midi, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# --- 2. Training & Inference Functions ---

def train_model(model, dataloader, epochs=50, lr=0.0001):
    """The main training loop. Runs on the GPU in Colab."""
    print(f"🚀 Starting training on {device}...")
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=84)

    for epoch in range(epochs):
        total_loss = 0
        for midi_batch, target_batch in dataloader:
            midi_batch, target_batch = midi_batch.to(
                device), target_batch.to(device)
            midi_batch, target_batch = midi_batch.t(), target_batch.t()

            optimizer.zero_grad()
            predictions = model(midi_batch)

            predictions = predictions.view(-1, NUM_BASS_POSITIONS)
            target_batch = target_batch.reshape(-1)

            loss = criterion(predictions, target_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    print("✅ Training complete!")
    return model


def map_notes_with_transformer(notes: list[dict]) -> list[dict]:
    """Inference wrapper: Translates AI numerical predictions back to Strings/Frets."""
    if not notes:
        return []

    model = BassTabTransformer().to(device)
    model.eval()

    midi_sequence = [librosa.note_to_midi(n['note']) for n in notes]
    src_tensor = torch.tensor(
        midi_sequence, dtype=torch.long).unsqueeze(1).to(device)

    with torch.no_grad():
        predictions = model(src_tensor)

    best_positions = predictions.argmax(dim=-1).squeeze(1).tolist()
    fretted_notes = []

    for i, note_data in enumerate(notes):
        predicted_class = best_positions[i]

        if predicted_class >= (NUM_STRINGS * POSITIONS_PER_STRING):
            string_name, fret_num = 'E', 0
        else:
            string_idx = predicted_class // POSITIONS_PER_STRING
            fret_num = predicted_class % POSITIONS_PER_STRING
            string_name = STRINGS[string_idx]

        note_data['string'] = string_name
        note_data['fret'] = fret_num
        fretted_notes.append(note_data)

    return fretted_notes


# --- 3. Test Execution Block (Run this in Colab to verify architecture) ---
if __name__ == "__main__":
    print(f"🖥️ System Check: Device is {device}")

    # Generate 100 fake "songs" to test if the model can learn
    # E.g., MIDI 40 always equals Position 12 (A string, 7th fret)
    dummy_midi = [[40, 42, 43, 45, 47] for _ in range(100)]
    dummy_targets = [[12, 14, 15, 17, 19] for _ in range(100)]

    test_dataset = BassTabDataset(dummy_midi, dummy_targets, max_length=10)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    my_transformer = BassTabTransformer()

    # Train for 50 epochs. You should see the Loss drop rapidly towards 0!
    trained_model = train_model(
        my_transformer, test_dataloader, epochs=50, lr=0.001)
