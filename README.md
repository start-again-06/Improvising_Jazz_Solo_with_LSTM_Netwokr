# Music Generation with LSTM
A deep learning project that generates music sequences using a many-to-many LSTM architecture trained on MIDI data. The model predicts notes and chords step-by-step, enabling the creation of new musical sequences.

## Dataset & Preprocessing
- Loads data from `original_metheny.mid`  
- Extracts sequences of notes/chords and encodes them as one-hot vectors  
- Total unique note values: `n_values = 90`  

## Project Pipeline

### Sequence Extraction
- Extract sequences of notes/chords from MIDI files.
- Encode sequences as one-hot vectors for model input.

### Model Construction
- Many-to-many LSTM architecture with 64-unit LSTM cells.
- Dense layer with softmax over note categories.
- Input shape: `(m, Tx, n_values)` with `Tx=30` time steps by default.

### Training
- Optimizer: Adam with decay
- Loss: Categorical crossentropy
- Trains for 100 epochs on sequences

## Inference
- Generate music step-by-step using previous outputs as input.
- Prediction loop over `Ty=50` time steps by default.

## Audio Conversion
- Convert predicted indices to MIDI using `generate_music`.
- Save MIDI to `output/my_music.midi` and convert to WAV.

## Dependencies
- TensorFlow
- Keras
- NumPy
- music21 library for symbolic music processing

## Folder Structure
- `data/` – Original MIDI files
- `outputs/` – Generated MIDI and WAV files
- `grammar.py` – Music grammar helpers
- `preprocess.py` – Data preprocessing logic
- `music_utils.py` – Audio and MIDI conversion
- `data_utils.py` – Sequence loading and encoding
- `qa.py` – Quality assurance utilities
- `outputs.py` – Output summary comparators
- `test_utils.py` – Unit test utilities

## Output
Generated music sequences in MIDI and WAV format. The model can produce raw sequences, post-training outputs, or fully generated compositions.

## Applications
- Symbolic music generation and composition
- Experimenting with sequence modeling for music
- Music-assisted creative projects or tools

## License
Intended for research, educational, and internal development use. For production or commercial deployment, additional testing and validation are recommended.


