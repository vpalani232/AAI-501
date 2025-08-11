 Gender Recognition from Audio

This project trains machine learning and deep learning models to classify a speaker's gender from audio recordings.  
It uses the **LibriSpeech ASR corpus** as the dataset and includes both **data preparation** and **model training** pipelines.

---

## 📂 Repository Structure

```
.
├── data_preparation.py                # Script for preparing audio dataset from LibriSpeech
├── Gender_recognition_from_Audio_final.ipynb  # Jupyter notebook for feature extraction, model training & evaluation
└── README.md                          # Project documentation
```

---

## 📊 Dataset

- **Source:** [LibriSpeech ASR corpus](https://www.openslr.org/12)
- **Subset Used:** `dev-clean` (can be replaced with other subsets)
- **Format:** Original `.flac` files are converted to `.wav` and then processed.
- **Labels:** Extracted from `SPEAKERS.TXT` mapping file.
- **Classes:** `male` and `female`

---

## ⚙️ Data Preparation

The `data_preparation.py` script:

1. Reads the `SPEAKERS.TXT` mapping file.
2. Extracts speaker gender information.
3. Copies `.flac` files into separate `male/` and `female/` folders.
4. Converts `.flac` files to `.wav` format.
5. Prepares the final dataset in:
   ```
   LibriSpeech/voice_dataset/
   ├── male/
   └── female/
   ```

**Usage:**
```bash
python data_preparation.py
```
Ensure that:
- `MAPPING_FILE` path is set to `SPEAKERS.TXT`
- `SOURCE_DIR` points to the folder containing audio files (e.g., `dev-clean`)

---

## 🤖 Model Training & Evaluation

The `Gender_recognition_from_Audio_final.ipynb` notebook includes:

### 1. Feature Extraction
- Converts audio to mel-spectrograms.
- Pads/clips audio to a fixed length (3 seconds).
- Uses **Librosa** for signal processing.

### 2. Models Implemented
- **Baseline:** Random Forest classifier.
- **Advanced Models:**
  - 1D CNN for spectrogram classification.
  - LSTM + CNN hybrid model for temporal and spectral feature learning.

### 3. Evaluation
- Accuracy & loss curves.
- Confusion matrix.
- Per-class precision, recall, and F1-score.

---

## 🛠 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

**requirements.txt** (example):
```
numpy
pandas
librosa
matplotlib
seaborn
tensorflow
scikit-learn
tqdm
pydub
```

---

## 🚀 Running the Project

1. **Prepare Dataset:**
   ```bash
   python data_preparation.py
   ```
2. **Open Notebook:**
   ```bash
   jupyter notebook Gender_recognition_from_Audio_final.ipynb
   ```
3. **Train Models:**
   - Run the notebook cells in order.

---

## 📌 Notes & Limitations

- Dataset used is **clean and recorded in controlled environments**.  
  In real-world conditions (background noise, poor microphones, accents), performance may drop.
- Model is trained on short audio clips (3 seconds).
- Can be extended to multi-class classification for age or emotion detection.

---

## 📄 License

This project is licensed under the MIT License.

---
