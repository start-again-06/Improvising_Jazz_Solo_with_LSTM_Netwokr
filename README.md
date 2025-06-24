## 🎶 Music Generation with LSTM

This module generates music sequences using a many-to-many LSTM architecture trained on MIDI data.

---

### 🎼 Dataset & Preprocessing

- Loads data from `original_metheny.mid`
- Preprocessing extracts sequences of notes/chords and encodes them as one-hot vectors
- Total unique note values: `n_values = 90`

```python
X, Y, n_values, indices_values, chords = load_music_utils('data/original_metheny.mid')
```

---

### 🧱 Model Architecture

Built using TensorFlow and Keras:

- Input shape: `(m, Tx, n_values)`
- LSTM Cell: 64 units
- Output Dense layer with softmax over note categories
- Repeats LSTM for `Tx` time steps (30 by default)

```python
model = djmodel(Tx=30, LSTM_cell=LSTM_cell, densor=densor, reshaper=reshaper)
```

---

### 🏋️ Training

- Optimizer: Adam with decay
- Loss: Categorical crossentropy
- Trains for 100 epochs on sequences

```python
model.compile(optimizer=opt, loss='categorical_crossentropy')
history = model.fit([X, a0, c0], list(Y), epochs=100)
```

---

### 🔮 Inference Model

- Generates music step-by-step using previous outputs as input
- Prediction loop over `Ty` steps (default = 50)

```python
inference_model = music_inference_model(LSTM_cell, densor, Ty=50)
results, indices = predict_and_sample(inference_model)
```

---

### 🎧 Audio Output

- Converts sampled indices to MIDI using `generate_music`
- Writes to `output/my_music.midi` and converts to WAV
- You can listen to:
  - Raw input: `30s_seq.wav`
  - Post-training: `30s_trained_model.wav`
  - Your model: `rendered.wav`

```python
out_stream = generate_music(inference_model, indices_values, chords)
mid2wav('output/my_music.midi')
```

---

### 🧪 Files Used

```bash
├── grammar.py           # Music grammar helpers
├── qa.py                # Quality assurance utilities
├── preprocess.py        # Data preparation logic
├── music_utils.py       # Audio and MIDI conversion
├── data_utils.py        # Sequence loading and encoding
├── outputs.py           # Output summary comparators
├── test_utils.py        # Unit test utilities
```

---

### 📚 References

- [music21 Library](http://web.mit.edu/music21/) – Symbolic music processing
- [Karpathy’s Char-RNN Inspiration](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
