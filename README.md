````markdown
# MYOCR — Handwritten Text Recognition with TrOCR

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

A fine-tuned Handwritten Text Recognition (HTR) system built using Microsoft's **TrOCR (Transformer-based OCR)** model. The model was fine-tuned on the **IAM Handwriting Line Dataset** and published on Hugging Face Hub.

The objective of this project is to recognize unconstrained handwritten text from line images and transcribe it into machine-readable text with high accuracy.

---

# Hugging Face Model

Model Repository:

```text
https://huggingface.co/june168/trocr-handwritten-iam
```

Model Name

```python
june168/trocr-handwritten-iam
```

Base Model

```python
microsoft/trocr-large-handwritten
```

---

# Table of Contents

- Objective
- Problem Statement
- Project Workflow
- Dataset
- Exploratory Data Analysis
- Preprocessing
- Model Architecture
- Training Configuration
- Evaluation
- Results
- Hugging Face Deployment
- Project Structure
- Installation
- Inference
- Requirements
- References


---

# Objective

The primary objective of this project is to develop an end-to-end handwritten text recognition system capable of converting handwritten text images into digital text.

The system should:

• Accept grayscale or RGB handwritten line images

• Generate corresponding text sequences

• Utilize transfer learning from a large-scale pre-trained transformer model

• Minimize preprocessing requirements

• Eliminate explicit character segmentation

• Demonstrate practical deployment through Hugging Face Hub


---

# Problem Statement


Handwritten Text Recognition (HTR) remains a challenging task in computer vision and natural language processing because handwriting exhibits significant variability.

Several issues commonly affect traditional OCR systems:


### Writer Variability

Different individuals write characters differently.


Examples include:

• Slant differences

• Character spacing

• Pen pressure

• Writing styles



---

### Connected Characters


Characters frequently overlap or connect together.


Examples:


```text
"hello"

may appear as


"h e l l o"

or

"hell o"

or

"hello"
```


Traditional segmentation-based approaches often fail.


---

### Noise


Scanned documents may contain:


• Blur

• Background artifacts

• Shadows

• Low contrast


---

### Language Understanding


OCR systems need to learn:


Visual Features


AND


Language Patterns


Example:


Image:


```text
Ths is an exmple
```


Desired output:


```text
This is an example
```



Language context improves transcription quality.


---

## Proposed Solution


This project leverages TrOCR, a Transformer-based Vision Encoder–Decoder architecture developed by Microsoft Research.


Advantages include:


✔ No character segmentation


✔ End-to-end learning


✔ Attention mechanism


✔ Context-aware decoding


✔ Transfer learning


✔ State-of-the-art OCR performance



---

# Project Workflow


```text

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
      ├── AdamW
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
      ├── Push to HF Hub
      └── Public Inference




```

---

# Dataset


## IAM Handwriting Database


Dataset:


```python
Teklia/IAM-line
```



Dataset Source


```text
https://huggingface.co/datasets/Teklia/IAM-line
```



IAM is considered one of the most widely used benchmark datasets for handwritten text recognition research.



Characteristics


| Property | Value |
|---------|-------|
| Language | English |
| Annotation Level | Text Line |
| Format | PNG Images |
| Labels | Ground Truth Text |
| Train Size | ~167 MB |
| Validation Size | ~25 MB |



Each sample contains:


### Input


Handwritten text image


### Target


Ground-truth transcription



Example


```text
Image:


"The quick brown fox"



Label:


The quick brown fox
```



---

# Exploratory Data Analysis


EDA was conducted to better understand the dataset characteristics before training.


## Sample Visualization


Random samples from the training split were visualized.


Goals:


• Verify labels


• Check image quality


• Inspect writing styles



---

## Image Statistics


Measured:


Minimum Width


Maximum Width


Average Width


Minimum Height


Maximum Height


Average Height



---

## Text Statistics


Calculated:


Character counts


Word counts


Sentence lengths


Average line length



---

## Character Frequency Analysis


Frequency distribution of:


Uppercase letters


Lowercase letters


Digits


Punctuation marks


Special symbols



---

## Visualization


Plots generated:


• Histogram of line lengths


• Histogram of word counts


• Character frequency bar chart


• Image dimension distribution


These analyses helped validate dataset consistency and informed preprocessing decisions.


---
````
