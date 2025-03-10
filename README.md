# Text-to-Speech using HuBERT Model

## Overview
This project implements a **Text-to-Speech (TTS) system** using the **HuBERT (Hidden-Unit BERT) Model**, a self-supervised learning approach designed for speech representation learning. The system converts text input into high-quality speech audio, leveraging HuBERT's powerful pre-trained model for natural-sounding speech synthesis.

## Features
- Converts input text into natural-sounding speech.
- Uses the **HuBERT model** for high-quality audio synthesis.
- Supports multiple languages and accents (if fine-tuned accordingly).
- Efficient and optimized inference for real-time applications.

## Methodology
1. **Text Preprocessing**: Cleans and normalizes text input.
2. **Phoneme Conversion**: Converts text into phonetic representations.
3. **HuBERT Model Processing**: Uses HuBERT embeddings for speech synthesis.
4. **Waveform Generation**: Converts model outputs into speech waveforms.
5. **Post-processing**: Enhances speech quality with noise reduction and prosody adjustments.

## Model/Algorithms Used
- **HuBERT (Hidden-Unit BERT)**: Self-supervised speech model.
- **Tacotron or Vocoder models**: For waveform synthesis (e.g., WaveGlow, HiFi-GAN).

## Implementation
1. **Install Dependencies:**
   ```bash
   pip install torch torchaudio transformers librosa numpy
   ```
2. **Load the HuBERT Model:**
   ```python
   from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
   import torchaudio

   model_name = "facebook/hubert-large-ls960-ft"
   processor = Wav2Vec2Processor.from_pretrained(model_name)
   model = Wav2Vec2ForCTC.from_pretrained(model_name)
   ```
3. **Convert Text to Speech:** (Requires fine-tuned TTS model)
   ```python
   text = "Hello, welcome to the Text-to-Speech system using HuBERT."
   # Implement phoneme conversion and speech synthesis pipeline
   ```

## Dataset
- **Common Voice**, **LJSpeech**, or **custom datasets** for training/fine-tuning.
- Preprocessed text-speech pairs.

## Evaluation
- **Mean Opinion Score (MOS)** for quality assessment.
- **Word Error Rate (WER)** for intelligibility.
- **Spectrogram Analysis** for waveform quality.

## IDE & Libraries
- **IDE:** Jupyter Notebook / VS Code / PyCharm
- **Libraries:** PyTorch, Torchaudio, Transformers, Librosa, NumPy

## Hyperparameter Tuning & Validation
- **Fine-tuning the model** with domain-specific speech datasets.
- **Validation using speech intelligibility metrics**.

## Future Enhancements
- Support for multiple languages.
- Improved prosody and naturalness.
- Real-time speech synthesis optimization.

## References
- [HuBERT: Self-Supervised Speech Representation Learning](https://arxiv.org/abs/2106.07447)
- [Facebook’s HuBERT Model](https://huggingface.co/facebook/hubert-large-ls960-ft)

## Author
Mahankali N V R Pavan Sai Mohan


