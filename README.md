```markdown
# Handwriting Signature Writer Identification using CNN

This project implements a Convolutional Neural Network (CNN) model to identify the writer of a given handwritten signature. The model is trained on a dataset of genuine and forged signatures and aims to classify a new signature sample to its respective writer.

This project was developed in a Google Colab environment utilizing TensorFlow and Keras.

![Example Prediction Output](images/example_prediction.png) <!-- REPLACE with a screenshot of your prediction results -->

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

## Project Overview

The primary goal of this project is to build a system capable of identifying the author of a handwritten signature. This has applications in various fields, including document verification and forensics. A CNN model is employed due to its effectiveness in image recognition tasks.

## Dataset

The dataset used for this project consists of handwritten signatures from multiple writers. It includes both genuine (`full_org`) and forged (`full_forg`) signatures.

- **Source:** [Specify the origin of your dataset. If public, name it and provide a link. If personal/self-collected, briefly describe it.]
- **Structure:** The data is organized into folders, typically with filenames indicating the writer ID (e.g., `original_58_1.png`, `forgeries_1_10.png`).
- **Image Format:** Primarily PNG, TIFF, or JPG files.

**Important Note on Data Privacy:** [If you are using real signature data, add a note about privacy and how you've handled/anonymized the data. Example: "All signature data used in this project has been anonymized and is used solely for research and educational purposes. If using your own dataset, ensure you have appropriate consent."].

## Methodology

### Data Preprocessing

1.  **Data Loading:** Image paths are collected from the specified directories on Google Drive.
2.  **Label Extraction:** Writer IDs are extracted from filenames to serve as labels.
3.  **Label Encoding:** String-based writer IDs are converted to integer labels.
4.  **Data Splitting:** The dataset is split into training and validation sets (e.g., 80% train, 20% validation) using `train_test_split` with stratification to maintain class proportions.
5.  **Image Preprocessing Function (`load_and_preprocess_image`):**
    *   Images are read and decoded.
    *   Resized to a standard input size of `64x256` pixels.
    *   Converted to grayscale (1 channel).
    *   Pixel values are normalized to the range `[0, 1]`.
6.  **`tf.data.Dataset` Pipeline:**
    *   Efficient data loading pipelines are created for training and validation using `tf.data.Dataset`.
    *   This includes mapping the preprocessing function, filtering out corrupted images, shuffling (for training), batching, and prefetching.

### Model Architecture

A Convolutional Neural Network (CNN) is constructed sequentially:

1.  **Conv2D Layer:** 32 filters, kernel size (3,3), ReLU activation, input shape (64, 256, 1).
2.  **MaxPooling2D Layer:** Pool size (2,2).
3.  **Conv2D Layer:** 64 filters, kernel size (3,3), ReLU activation.
4.  **MaxPooling2D Layer:** Pool size (2,2).
5.  **Conv2D Layer:** 128 filters, kernel size (3,3), ReLU activation.
6.  **MaxPooling2D Layer:** Pool size (2,2).
7.  **Flatten Layer:** To convert 2D feature maps into a 1D vector.
8.  **Dense Layer:** 128 units, ReLU activation.
9.  **Dropout Layer:** Dropout rate of 0.5 for regularization to prevent overfitting.
10. **Output Dense Layer:** `num_classes` units (equal to the number of unique writers), no activation function (as `from_logits=True` is used in the loss function).

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 62, 254, 32)       320

 max_pooling2d (MaxPooling2  (None, 31, 127, 32)       0
 D)

 conv2d_1 (Conv2D)           (None, 29, 125, 64)       18496

 max_pooling2d_1 (MaxPoolin  (None, 14, 62, 64)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 12, 60, 128)       73856

 max_pooling2d_2 (MaxPoolin  (None, 6, 30, 128)        0
 g2D)

 flatten (Flatten)           (None, 23040)             0

 dense (Dense)               (None, 128)               2949248

 dropout (Dropout)           (None, 128)               0

 dense_1 (Dense)             (None, [NUM_CLASSES])          [PARAMS_FOR_LAST_LAYER]

=================================================================
Total params: [TOTAL_PARAMS]
Trainable params: [TRAINABLE_PARAMS]
Non-trainable params: 0
_________________________________________________________________
```
*(Replace `[NUM_CLASSES]`, `[PARAMS_FOR_LAST_LAYER]`, `[TOTAL_PARAMS]`, `[TRAINABLE_PARAMS]` with actual values from your `model.summary()` output)*

### Training

-   **Optimizer:** Adam (`adam`).
-   **Loss Function:** `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)`. Suitable for multi-class classification with integer labels and model outputting logits.
-   **Metrics:** Accuracy (`accuracy`).
-   **Epochs:** The model was trained for [Number, e.g., 10 or 50] epochs. (Adjust based on your final training).
-   The training history (accuracy and loss for training and validation sets over epochs) is recorded.

## Results

After training, the model achieved the following performance on the validation set:

-   **Validation Accuracy:** [Your_Validation_Accuracy, e.g., 95.83]%
-   **Validation Loss:** [Your_Validation_Loss]

The training progress can be visualized through the following plots:

**Training and Validation Accuracy:**
![Training and Validation Accuracy Plot](images/accuracy_plot.png) <!-- REPLACE with an image of your accuracy plot -->

**Training and Validation Loss:**
![Training and Validation Loss Plot](images/loss_plot.png) <!-- REPLACE with an image of your loss plot -->

The model demonstrates good performance in identifying the correct writer for a given signature. Further testing on unseen data (as shown in cells 12, 13, and 14 of the notebook) confirms its capabilities. For example, when testing all signatures for writer ID '58', an overall accuracy of [Accuracy_For_Writer_58, e.g., 95.83]% ([Correct_Count]/[Total_Count]) was achieved.

## How to Use

### Prerequisites

-   Google Colaboratory environment (or a local environment with Python and necessary libraries).
-   TensorFlow `[Your_TF_Version, e.g., 2.x]`
-   Matplotlib
-   NumPy
-   Scikit-learn
-   Pillow (PIL)
-   Access to Google Drive (if using data stored on Drive).

### Setup

1.  **Clone the repository (optional, if running locally):**
    ```bash
    git clone https://github.com/TranHuuDat2004/handwriting-signature-recognition.git
    cd handwriting-signature-recognition
    ```
2.  **Upload the Notebook:** Open `[Your_Notebook_Name].ipynb` in Google Colab.
3.  **Prepare Data:**
    *   Ensure your signature dataset is uploaded to Google Drive.
    *   **Crucially, update the `BASE_DATA_DIR` variable in Cell #2 of the notebook** to point to the correct path of your `SIGNATURES` directory on Google Drive.
        ```python
        BASE_DATA_DIR = '/content/drive/MyDrive/YOUR_PATH_TO/SIGNATURES' # <<=== MODIFY THIS
        ```
    *   The dataset should have subfolders named `full_org` (for genuine signatures) and `full_forg` (for forged signatures), with image filenames following a convention that includes the writer ID (e.g., `original_{writer_id}_{sample_num}.png`).

### Running the Notebook

1.  **Mount Google Drive:** Run Cell #2 to mount your Google Drive. You will be prompted for authorization.
2.  **Execute Cells Sequentially:** Run each cell in the notebook from top to bottom.
    *   Cell #3: Collects image paths and extracts writer IDs.
    *   Cell #4: Processes labels and splits data.
    *   Cell #5 & #6: Defines preprocessing and creates `tf.data.Dataset` pipelines.
    *   Cell #7 & #8: Builds and compiles the CNN model.
    *   Cell #9: Trains the model. This step may take some time depending on the dataset size and Colab resources.
    *   Cell #10: Evaluates the model and plots training history.
    *   Cell #11: Saves the trained model to your Google Drive (in `.keras` and `.h5` formats).
    *   Cell #12: Demonstrates how to predict on a single new signature image (update the `new_signature_path`).
    *   Cell #13: Shows how to perform batch predictions on a sample of images.
    *   Cell #14: Allows testing all signatures for a specific `TARGET_WRITER_ID_STR` (update this ID).

## Future Work

-   Experiment with different CNN architectures (e.g., ResNet, VGG, MobileNet) or attention mechanisms.
-   Explore more advanced data augmentation techniques.
-   Implement a Siamese Network architecture for one-shot signature verification (verifying if two signatures are from the same person, rather than identifying the writer from a known set).
-   Develop a user interface (e.g., a web application) for easier interaction with the model.
-   Train on a larger and more diverse dataset to improve generalization.

## Author

-   **Tran Huu Dat**
    -   GitHub: [@TranHuuDat2004](https://github.com/TranHuuDat2004)
    -   [Link_To_Your_Portfolio_Or_LinkedIn (Optional)]

## Acknowledgments

-   [Mention any datasets, papers, or individuals that inspired or helped your work, if applicable.]
-   TensorFlow and Keras teams for their excellent libraries.
-   Google Colab for providing a free and accessible environment for deep learning.
```