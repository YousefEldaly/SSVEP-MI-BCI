# High-Performance EEG Motor Imagery Classification (0.71 F1-Score)

This repository contains the training and inference scripts for a high-performing deep learning model designed to classify Motor Imagery (MI) tasks from EEG data. The model achieves a Macro F1-score of 0.71 by combining robust preprocessing, powerful feature extraction using Common Spatial Patterns (CSP), and a tailored 1D Convolutional Neural Network (CNN) architecture.

## Methodology

The success of this model is built on a three-stage pipeline: meticulous preprocessing, discriminative feature extraction, and a carefully designed neural network.

### 1. Data Preprocessing

The preprocessing pipeline is designed to clean the raw EEG signals and standardize them for optimal model performance.

- **Data Filtering**: The process begins by excluding noisy or unreliable trials. A predefined list of subjects and their corresponding trial sessions are dropped to ensure data quality.
- **Channel Selection**: To focus on the most relevant brain regions for motor imagery, only data from the motor cortex channels **C3, CZ, and C4** are used for the analysis.
- **Trial-wise Standardization**: Each trial is processed independently. A `StandardScaler` is applied to each trial's data to normalize its distribution (zero mean, unit variance). This step is critical for making the model robust to variations in signal amplitude between different trials and subjects.
- **Data Augmentation**: To improve generalization and prevent overfitting, a simple yet effective data augmentation technique is applied exclusively to the training data. A small amount of Gaussian noise is added to the signal, creating new, slightly different training examples for the model to learn from.

### 2. Feature Extraction: Common Spatial Patterns (CSP)

Instead of feeding the raw time-series data directly into the deep learning model, we first use the **Common Spatial Patterns (CSP)** algorithm. CSP is a highly effective feature extraction technique specifically for discriminating between two classes in EEG data.

- **Goal**: The objective of CSP is to learn a set of spatial filters that maximize the variance of the signal for one class while minimizing it for the other. This process creates new, transformed signals that are highly discriminative.
- **Implementation**:
    1.  The `mne.decoding.CSP` transformer is initialized to find the **2 most discriminative components** (`n_components=2`).
    2.  The CSP object is `fit` only on the preprocessed **training data** and corresponding labels (`X_train`, `y_train`). This step is crucial to prevent data leakage from the validation or test sets.
    3.  The fitted CSP transformer is then used to `transform` the training, validation, and test datasets. This projects the original multi-channel EEG data into a new 2-dimensional CSP space, where the resulting signals have enhanced class-separability.
    4.  Finally, the data is transposed to match the input shape expected by the 1D CNN: `(n_trials, n_samples, n_csp_components)`.

### 3. Model Architecture

The core of the classification pipeline is a 1D Convolutional Neural Network (CNN) designed to learn temporal patterns from the CSP-transformed features.

The model is a `Sequential` stack of layers:

| Layer | Type | Details | Purpose |
| :--- | :--- | :--- | :--- |
| 1 | **Conv1D** | 32 filters, kernel size=10, activation='relu' | Learns low-level temporal features from the CSP components. |
| 2 | **BatchNormalization** | - | Stabilizes learning and reduces internal covariate shift. |
| 3 | **MaxPooling1D** | pool size=3 | Downsamples the feature maps, reducing dimensionality and providing translational invariance. |
| 4 | **Conv1D** | 64 filters, kernel size=10, activation='relu' | Learns more complex, higher-level features from the previous layer's output. |
| 5 | **BatchNormalization** | - | Stabilizes learning. |
| 6 | **MaxPooling1D** | pool size=3 | Further downsamples the feature maps. |
| 7 | **Conv1D** | 128 filters, kernel size=10, activation='relu' | Learns even more abstract and complex temporal patterns. |
| 8 | **BatchNormalization** | - | Stabilizes learning. |
| 9 | **MaxPooling1D** | pool size=3 | Further downsamples the feature maps. |
| 10 | **Flatten** | - | Converts the 2D feature maps into a 1D vector for the fully connected layers. |
| 11 | **Dense** | 100 neurons, activation='relu' | A fully connected layer that performs high-level feature combination. |
| 12 | **Dropout** | rate=0.5 | Regularization technique to prevent overfitting by randomly setting 50% of neuron activations to zero during training. |
| 13 | **Dense (Output)** | 1 neuron, activation='sigmoid' | The final output layer that produces a probability score for the binary classification task. |

The model is compiled with the `adam` optimizer and `binary_crossentropy` loss function, which are standard choices for binary classification tasks.

## How to Use

### 1. Training the Model

The `training_script.py` file handles the entire process of loading data, preprocessing, fitting the CSP transformer, and training the CNN model.

**To run training:**
```bash
python training_script.py