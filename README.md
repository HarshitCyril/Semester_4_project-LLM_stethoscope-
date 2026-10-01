# LLM Stethoscope 🩺

An AI diagnostic tool that converts acoustic respiratory sounds into audio spectrograms to classify lung conditions using Vision Transformers.

## Overview
Respiratory sound analysis is often limited by background noise and subjective evaluation. The LLM Stethoscope processes real-time chest auscultation audio, converts raw signals into high-density spectrograms, and feeds them into a fine-tuned Vision Transformer (ViT) to assist in preliminary diagnostic feedback.

## What It Does
- **Signal Processing:** Filters ambient interference and converts continuous audio into structured spectrograms using Librosa.
- **Pattern Classification:** Applies Vision Transformer architectures to identify acoustic anomalies in pulmonary audio.
- **Noise Reduction:** Implements custom digital signal processing filters to clean audio feeds before inference.
- **Fast Feedback:** Keeps inference latency under 150ms for near-real-time clinical support.

## Tech Stack
- **Language:** Python
- **ML / Deep Learning:** PyTorch, Torchvision, Hugging Face
- **Audio Processing:** Librosa, SciPy, NumPy
- **Deployment:** Docker, FastAPI

## Quickstart

```bash
# Clone the repository
git clone [https://github.com/HarshitCyril/LLM-Stethoscope.git](https://github.com/HarshitCyril/LLM-Stethoscope.git)
cd LLM-Stethoscope

# Install dependencies
pip install -r requirements.txt

# Run inference on a sample audio file
python main.py --input sample_audio.wav
