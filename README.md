# MYOCR — Handwritten Text Recognition with TrOCR

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

A fine-tuned Handwritten Text Recognition (HTR) system built using Microsoft's **TrOCR (Transformer-based OCR)** model. The model was fine-tuned on the **IAM Handwriting Line Dataset** and published on Hugging Face Hub.

The objective of this project is to recognize unconstrained handwritten text from line images and transcribe it into machine-readable text with high accuracy.

---

## Hugging Face Model

Model Repository:
```
https://huggingface.co/june168/trocr-handwritten-iam
```

Model Name:
```
june168/trocr-handwritten-iam
```

Base Model:
```
microsoft/trocr-large-handwritten
```

---

## Table of Contents

- [Objective](#objective)
- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Evaluation](#evaluation)
- [Results](#results)
- [Hugging Face Deployment](#hugging-face-deployment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Inference](#inference)
- [Requirements](#requirements)
- [References](#references)

---

## Objective

The primary objective of this project is to develop an end-to-end handwritten text recognition system capable of converting handwritten text images into digital text.

The system should:

- Accept grayscale or RGB handwritten line images
- Generate corresponding text sequences
- Utilize transfer learning from a large-scale pre-trained transformer model
- Minimize preprocessing requirements
- Eliminate explicit character segmentation
- Demonstrate practical deployment through Hugging Face Hub

---

## Problem Statement

Handwritten Text Recognition (HTR) remains a challenging task in computer vision and natural language processing because handwriting exhibits significant variability.

Several issues commonly affect traditional OCR systems:

### Writer Variability

Different individuals write characters differently. Examples include:

- Slant differences
- Character spacing
- Pen pressure
- Writing styles

### Connected Characters

Characters frequently overlap or connect together. Traditional segmentation-based approaches often fail when text appears as:

```
"hello"  →  "h e l l o"  or  "hell o"  or  "hello"
```

### Noise

Scanned documents may contain:

- Blur
- Background artifacts
- Shadows
- Low contrast

### Language Understanding

OCR systems need to learn both visual features and language patterns simultaneously.

```
Image input:   "Ths is an exmple"
Desired output: "This is an example"
```

Language context improves transcription quality significantly.

### Proposed Solution

This project leverages TrOCR, a Transformer-based Vision Encoder–Decoder architecture developed by Microsoft Research.

Advantages include:

- ✔ No character segmentation required
- ✔ End-to-end learning
- ✔ Attention mechanism
- ✔ Context-aware decoding
- ✔ Transfer learning from large-scale pre-training
- ✔ State-of-the-art OCR performance

---

## Project Workflow

```
Raw Dataset
     │
     ▼
1. Data Loading
     │
     ├── IAM-line Dataset
     ├── Train Split
     └── Validation Split
     │
     ▼
2. Exploratory Data Analysis
     │
     ├── Sample Visualization
     ├── Character Analysis
     ├── Width Distribution
     ├── Height Distribution
     └── Text Statistics
     │
     ▼
3. Preprocessing
     │
     ├── Resize Images
     ├── RGB Conversion
     ├── Normalization
     └── Tokenization
     │
     ▼
4. Model Setup
     │
     ├── TrOCR Processor
     ├── VisionEncoderDecoderModel
     └── CUDA Initialization
     │
     ▼
5. Fine-Tuning
     │
     ├── Dataset Class
     ├── DataLoader
     ├── AdamW Optimizer
     ├── Training Loop
     └── Validation Loop
     │
     ▼
6. Evaluation
     │
     ├── Loss Curve
     ├── Prediction Samples
     └── Qualitative Assessment
     │
     ▼
7. Deployment
     │
     ├── Save Model
     ├── Push to HuggingFace Hub
     └── Public Inference
```

---

## Dataset

### IAM Handwriting Database

```python
dataset = load_dataset("Teklia/IAM-line")
```

Dataset Source:
```
https://huggingface.co/datasets/Teklia/IAM-line
```

IAM is one of the most widely used benchmark datasets for handwritten text recognition research.

| Property | Value |
|----------|-------|
| Language | English |
| Annotation Level | Text Line |
| Format | PNG Images |
| Labels | Ground Truth Text |
| Train Size | ~167 MB |
| Validation Size | ~25 MB |

Each sample contains:

**Input** — Handwritten text line image

**Target** — Ground-truth transcription

Example:
```
Image:  [handwritten image of "The quick brown fox"]
Label:  The quick brown fox
```

---

## Exploratory Data Analysis

EDA was conducted to better understand the dataset characteristics before training.

### Sample Visualization

Random samples from the training split were visualized to:

- Verify label correctness
- Check image quality
- Inspect handwriting styles

### Image Statistics

Measured across the full training split:

| Metric | Value |
|--------|-------|
| Minimum Width | varies |
| Maximum Width | varies |
| Average Width | computed |
| Minimum Height | varies |
| Maximum Height | varies |
| Average Height | computed |

### Text Statistics

Calculated per line:

- Character counts per line
- Word counts per line
- Sentence lengths
- Average line length

### Character Frequency Analysis

Frequency distribution computed for:

- Uppercase letters (A–Z)
- Lowercase letters (a–z)
- Digits (0–9)
- Punctuation marks
- Special symbols

### Visualizations Generated

- Histogram of line lengths
- Histogram of word counts per line
- Character frequency bar chart (letters, digits, punctuation)
- Image dimension distribution

These analyses validated dataset consistency and informed preprocessing decisions.

---

## Preprocessing

| Step | Details |
|------|---------|
| Resize | 384 × 384 pixels (ViT encoder requirement) |
| Color conversion | Convert to RGB |
| Normalization | Mean = 0.5, Std = 0.5 per channel |
| Tokenization | RobertaTokenizer (BPE), vocab size 50,265 |
| Processor | TrOCRProcessor handles both image and text |

---

## Model Architecture

TrOCR is a Vision Encoder–Decoder model combining:

```
Input Image (384 × 384 × 3)
        │
        ▼
  ViT Encoder (Large)
  ├── 24 Transformer layers
  ├── Hidden size: 1024
  ├── 16 attention heads
  └── Patch size: 16 × 16
        │
  Encoded patch features
        │
        ▼
  TrOCR Decoder (BART-style)
  ├── 12 Transformer layers
  ├── Hidden size: 1024
  ├── 16 attention heads
  ├── Cross-attention over encoder outputs
  └── Max sequence length: 512
        │
        ▼
  Output tokens → Decoded text string
```

| Component | Details |
|-----------|---------|
| Encoder | ViT-Large (Vision Transformer) |
| Decoder | TrOCR / BART-style autoregressive decoder |
| Vocabulary size | 50,265 (RoBERTa BPE) |
| Parameters | ~558M |
| Model format | SafeTensors |

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Base model | `microsoft/trocr-large-handwritten` |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Batch size | variable |
| Device | GPU (CUDA) / CPU fallback |
| Loss function | Cross-entropy (built into VisionEncoderDecoderModel) |

---

## Evaluation

### Loss Curve

Training and validation loss were tracked per epoch and plotted using matplotlib to monitor convergence and detect overfitting.

### Qualitative Assessment

Predictions were compared against ground-truth labels on held-out validation images.

---

## Results

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | decreasing | decreasing |
| 2 | decreasing | decreasing |
| 3 | decreasing | decreasing |
| 4 | decreasing | decreasing |
| 5 | converged | converged |

### Sample Predictions

```
Image: has_its_premiere_tom.png
Ground truth:  "has its premiere tom"
Predicted:     "has its premiere tom"

Image: chair_The_seat_res.png
Ground truth:  "chair . The seat res"
Predicted:     "chair . The seat res"
```

---

## Hugging Face Deployment

The fine-tuned model is publicly available on Hugging Face Hub.

```
https://huggingface.co/june168/trocr-handwritten-iam
```

Load the model directly:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("june168/trocr-handwritten-iam")
model = VisionEncoderDecoderModel.from_pretrained("june168/trocr-handwritten-iam")
```

---

## Project Structure

```
OCR handWriten/
├── README.md                    # Project documentation
├── Text.ipynb                   # Main notebook: EDA, training, inference
└── MYOCR/
    └── content/
        └── MYOCR/               # Saved fine-tuned model
            ├── model.safetensors
            ├── config.json
            ├── generation_config.json
            ├── preprocessor_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── vocab.json
            ├── merges.txt
            └── special_tokens_map.json
```

---

## Installation

```bash
pip install transformers datasets torch torchvision pillow matplotlib seaborn
```

---

## Inference

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load from Hugging Face Hub
processor = TrOCRProcessor.from_pretrained("june168/trocr-handwritten-iam")
model = VisionEncoderDecoderModel.from_pretrained("june168/trocr-handwritten-iam")
model.to(device)
model.eval()

# Load your handwritten image
image = Image.open("your_handwritten_image.png").convert("RGB")

# Preprocess
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# Generate prediction
with torch.no_grad():
    generated_ids = model.generate(pixel_values)

predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Predicted:", predicted_text)
```

---

## Requirements

| Library | Purpose |
|---------|---------|
| `transformers` | TrOCR model, processor, tokenizer |
| `datasets` | Load IAM-line from HuggingFace |
| `torch` | Model training and inference |
| `Pillow` | Image loading and conversion |
| `numpy` | Array operations |
| `matplotlib` | Loss curve and EDA plots |
| `seaborn` | Character frequency visualization |

---

## References

- [TrOCR: Transformer-based OCR with Pre-trained Models](https://arxiv.org/abs/2109.10282) — Li et al., Microsoft Research
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [Teklia/IAM-line on HuggingFace](https://huggingface.co/datasets/Teklia/IAM-line)
- [microsoft/trocr-large-handwritten](https://huggingface.co/microsoft/trocr-large-handwritten)
- [june168/trocr-handwritten-iam](https://huggingface.co/june168/trocr-handwritten-iam)
