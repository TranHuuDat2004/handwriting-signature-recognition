# Handwriting Signature Writer Identification using CNN

This project implements a Convolutional Neural Network (CNN) model to identify the writer of a given handwritten signature. The model is trained on a dataset of genuine and forged signatures, and it aims to classify a new signature sample to its respective writer.

Developed in Google Colab using TensorFlow and Keras.

![Example Prediction Output](img/output.PNG) <!-- Replace with your result screenshot -->

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Architecture](#model-architecture)
    - [Training](#training)
- [Results](#results)
- [How to Use](#how-to-use)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
    - [Running the Notebook](#running-the-notebook)
- [Future Work](#future-work)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal is to build a system that identifies the author of a handwritten signature. This has applications in forensics and document verification. A CNN is used due to its effectiveness in image recognition tasks.

---

## Dataset

The dataset contains handwritten signatures from multiple writers, including both genuine (`full_org`) and forged (`full_forg`) samples.

- **Source:** [Add dataset name/link or brief explanation if private]
- **Structure:**
    - Example filenames: `original_58_1.png`, `forgeries_1_10.png`
- **Image Format:** PNG, TIFF, or JPG

> **Note on Privacy:** All data has been anonymized and used solely for research purposes. Ensure proper consent when using real signature data.

---

## Methodology

### Data Preprocessing

- **Loading:** Signature paths collected from Google Drive
- **Labeling:** Writer IDs extracted from filenames
- **Encoding:** String IDs converted to integers
- **Splitting:** 80/20 train-validation split (with stratification)
- **Image Processing:**
    - Resize to `64x256`
    - Grayscale (1 channel)
    - Normalize pixel values `[0, 1]`

> **TensorFlow Dataset Pipeline** is used for efficient loading, batching, and prefetching.

---

### Model Architecture

CNN model structure:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 62, 254, 32)       320
 max_pooling2d (MaxPooling2D) (None, 31, 127, 32)      0
 conv2d_1 (Conv2D)           (None, 29, 125, 64)       18496
 max_pooling2d_1 (MaxPooling2D) (None, 14, 62, 64)     0
 conv2d_2 (Conv2D)           (None, 12, 60, 128)       73856
 max_pooling2d_2 (MaxPooling2D) (None, 6, 30, 128)     0
 flatten (Flatten)           (None, 23040)             0
 dense (Dense)               (None, 128)               2949248
 dropout (Dropout)           (None, 128)               0
 dense_1 (Dense)             (None, 55)                7095
=================================================================
Total params: 3,046,015
Trainable params: 3,046,015
Non-trainable params: 0
_________________________________________________________________
```

---

### Training

- **Optimizer:** Adam
- **Loss:** `SparseCategoricalCrossentropy(from_logits=True)`
- **Metrics:** Accuracy
- **Epochs:** 30 (or adjust as needed)

Training history includes plots for accuracy and loss across epochs.

---

## Results

On validation set:

- **Validation Accuracy:** 95.83%
- **Validation Loss:** 0.25

**Training and Validation Accuracy:**  
![Accuracy Plot](img/accuracy.PNG)

**Training and Validation Loss:**  
![Loss Plot](img/loss.PNG)

**Example Test (Writer ID: 58):**

- **Accuracy:** 95.83% (23/24 correct)

---

## How to Use

### Prerequisites

- Google Colab or local Python environment
- Python Libraries:
    - TensorFlow (2.x)
    - NumPy
    - Matplotlib
    - scikit-learn
    - Pillow

### Setup

1. **Clone the repo (optional):**


        git clone https://github.com/TranHuuDat2004/handwriting-signature-recognition.git
        cd handwriting-signature-recognition


2. **Upload Notebook to Colab**

3. **Prepare Dataset:**
        - Upload dataset to Google Drive
        - Update `BASE_DATA_DIR` in the notebook:


            BASE_DATA_DIR = '/content/drive/MyDrive/YOUR_PATH_TO/SIGNATURES'


---

### Running the Notebook

1. **Mount Google Drive**
2. **Run cells sequentially:**
        - Collect paths and labels
        - Preprocess images
        - Create Dataset objects
        - Build and train the model
        - Evaluate and save model
        - Predict single or batch samples
        - Evaluate on a specific writer ID

---

## Future Work

- Try ResNet, VGG, or MobileNet
- Improve data augmentation
- Explore Siamese Networks for signature verification
- Build a web interface (e.g., with Streamlit)
- Train on larger datasets

---

## Author

**Tran Huu Dat**  
GitHub: [@TranHuuDat2004](https://github.com/TranHuuDat2004)  

---

## Acknowledgments

- TensorFlow and Keras for deep learning frameworks
- Google Colab for providing a free GPU environment
- [Your dataset source or research papers if applicable]

---
