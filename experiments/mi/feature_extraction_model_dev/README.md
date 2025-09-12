# Advancements in EEG Motor Imagery Classification

This document outlines the goals and results of a series of experiments focused on developing and refining models for classifying Electroencephalography (EEG) data from a Motor Imagery (MI) task. The experiments evolved from classical machine learning approaches to more complex hybrid deep learning architectures and culminated in rigorous, subject-independent validation.

## I. Feature Extraction Strategies

### Goal
The primary goal of feature extraction was to transform raw, noisy EEG signals into a more informative and lower-dimensional representation that machine learning models could effectively learn from. This involved two main approaches: manual feature engineering for classical models and automated feature learning through deep learning.

### Experiments & Results

#### **Common Spatial Patterns (CSP)**
- **Experiment:** CSP, a highly effective technique for MI tasks, was implemented to find spatial filters that maximize the variance between the "Left" and "Right" hand imagery classes. The log-variance of these spatially filtered signals was then used as a powerful feature for classification. This technique was a cornerstone for both classical machine learning models and as an input to some hybrid deep learning models.
- **Result:** The CSP-filtered data provided a strong baseline and proved to be a critical component in achieving high classification accuracy, especially when combined with models like LDA and SVM.

#### **End-to-End Feature Learning with CNNs**
- **Experiment:** Instead of manual feature extraction, several notebooks explored Convolutional Neural Networks (CNNs) to learn relevant features directly from the preprocessed time-series EEG data. These models used 1D convolutional layers to capture temporal dependencies in the signal.
- **Result:** The end-to-end models demonstrated the ability to learn discriminative patterns without relying on predefined features like band power or CSP, streamlining the pipeline.

#### **Advanced Handcrafted Features**
- **Experiment:** For hybrid models, a rich set of neurophysiologically-inspired features were engineered. These included:
    - **Band Power:** Power in the mu (8-12 Hz) and beta (13-30 Hz) bands.
    - **Lateralization Features:** Differences and asymmetry ratios in power between the C3 and C4 electrodes to capture the contralateral brain activity characteristic of motor imagery.
    - **Statistical Features:** Skewness and kurtosis.
    - **Hjorth Parameters:** Measures of signal activity, mobility, and complexity.
- **Result:** These handcrafted features, when combined with the learned features from the CNN branch in hybrid models, provided additional context that helped improve overall classification performance.

## II. Model Development and Architectures

### Goal
To build, train, and compare various classification models, ranging from established machine learning baselines to novel hybrid deep learning architectures, in order to identify the most effective approach for this dataset.

### Experiments & Results

#### **Classical Machine Learning Ensemble**
- **Experiment:** An initial exploration involved training and evaluating an ensemble of classical models, including Linear Discriminant Analysis (LDA), Support Vector Machine (SVM), and Random Forest (RF). These models were typically paired with features extracted via CSP.
- **Result:** This ensemble approach served as a robust baseline, demonstrating the effectiveness of traditional methods on this type of data and setting a benchmark for more complex models to surpass.

#### **Hybrid CNN + MLP/Logistic Regression Models**
- **Experiment:** The core of the development phase involved creating and testing various hybrid architectures. The common theme was to combine a CNN branch for learning from the raw EEG signal with a Multi-Layer Perceptron (MLP) or Logistic Regression branch that took in the handcrafted features.
    - One notable architecture used separate CNN branches to process signals from the left (C3) and right (C4) hemispheres before fusing them with the feature-based MLP branch.
- **Result:** These hybrid models consistently outperformed models that used either handcrafted features or raw EEG data alone. The combination of learned spatial-temporal features from the CNN and explicit neurophysiological features from the MLP proved highly effective, with one such model achieving a validation F1-score of **0.71**.

#### **Attention-Based CNN**
- **Experiment:** An attention mechanism was incorporated into a CNN architecture to allow the model to focus on the most informative time segments of the EEG signal.
- **Result:** This approach showed promise in improving the model's ability to discern relevant patterns in the time-series data.

#### **SVM with Riemannian Geometry**
- **Experiment:** This experiment explored a different paradigm by using Riemannian geometry to classify EEG signals. Covariance matrices of the EEG trials were used as features, and an SVM with a Riemannian kernel was used for classification.
- **Result:** This method provides a powerful alternative to traditional feature extraction and classification techniques, leveraging the geometric properties of the data.

### Feature and Model Comparison Table

| Feature Extraction Technique | Model Architecture | Goal | Approximate F1-Score |
| :--- | :--- | :--- | :--- |
| **Common Spatial Patterns (CSP)** | LDA, SVM, Random Forest | Establish a strong baseline using a proven feature extraction method for MI tasks. | ~0.65 |
| **Riemannian Geometry** | SVM with Riemannian Kernel | Classify trials based on their geometric properties (covariance matrices) instead of traditional features. | ~0.67 |
| **End-to-End Learning** | Standard CNN | Learn discriminative features directly from the raw, preprocessed EEG time-series data. | ~0.62 |
| **Handcrafted + Learned Features** | Hybrid CNN + MLP | Combine the power of automated feature learning (CNN) with domain-specific, handcrafted features (MLP). | **~0.71** |
| **End-to-End + Attention** | Attention-Based CNN | Improve the CNN by allowing it to dynamically focus on the most relevant temporal parts of the signal. | ~0.68 |

## III. Model Evaluation and Validation

### Goal
To rigorously evaluate the performance of the developed models using appropriate validation strategies and metrics to ensure the results were robust and generalizable.

### Experiments & Results

#### **Cross-Validation**
- **Experiment:** A 5-fold cross-validation strategy was implemented to provide a reliable estimate of model performance. This involved splitting the combined training and validation data into five folds and iteratively training on four while testing on the fifth, ensuring that every sample was used for validation exactly once.
- **Result:** The cross-validation results gave a more stable and trustworthy measure of how the models were likely to perform on unseen data, compared to a single train-validation split. The mean F1-scores across the folds for various models were calculated, providing a strong basis for model comparison.

#### **Leave-One-Subject-Out (LOSO) Cross-Validation**
- **Experiment:** To test the models' ability to generalize to new, unseen individuals, a Leave-One-Subject-Out (LOSО) cross-validation was performed. In this stringent method, the model is trained on data from all subjects except one, which is held out for testing. This process is repeated for every subject.
- **Result:** LOSО provides the most realistic estimate of real-world performance. The F1 scores from this evaluation highlight how well the models perform on users not included in the training set, which is critical for BCI applications.

#### **Systematic Model Benchmarking**
- **Experiment:** A dedicated notebook was created to systematically train and evaluate a batch of different model architectures on the fully preprocessed data. This provided a head-to-head comparison under identical data conditions.
- **Result:** This structured benchmark was essential for identifying the most promising models to carry forward for more intensive testing like LOSО and for final submission, ensuring that the selected model was chosen based on comprehensive evidence.

#### **Performance Metrics**
- **Metrics Used:** Beyond simple accuracy, the primary metric for evaluation was the **Macro F1-score**, which is crucial for datasets that might have class imbalance. Detailed classification reports and confusion matrices were also generated to analyze the models' performance on a per-class basis.
- **Result:** The focus on the F1-score and detailed reports allowed for a nuanced understanding of model strengths and weaknesses, particularly in how well they balanced precision and recall for both the 'Left' and 'Right' classes. One of the best-performing models demonstrated a significant improvement in the F1-score for the 'Right' class after optimization, achieving **0.71** overall.