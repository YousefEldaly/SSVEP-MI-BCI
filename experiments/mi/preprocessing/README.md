# Explorations in EEG Preprocessing for Motor Imagery Data

This document summarizes a series of experiments in preprocessing EEG data for a Motor Imagery (MI) task. The notebooks explore various techniques to handle artifacts, manage signal variability, and clean the data for subsequent analysis and machine learning.

## 1. Artifact Rejection Strategies

A primary focus was to identify and handle heavily contaminated trials. The core idea was that removing the noisiest data points or trials could lead to better model performance.

### Experiment 1: Threshold-Based Trial Dropping
- **Source Notebooks**: `preprocessing-using-threshold-25.ipynb`, `preprocessing-using-threshold-40-improved.ipynb`
- **Goal**: To establish a simple and effective rule for rejecting entire trials based on the proportion of artifact-contaminated samples within them.
- **Method**:
    - An "artifact mask" was generated for each trial, flagging individual samples affected by issues like high amplitude, signal clipping, or motion artifacts.
    - Trials were dropped if the percentage of these flagged samples exceeded a predefined threshold. Experiments were conducted with thresholds of 25% and 40%.
- **Findings**:
    - A stricter **25% threshold** led to the removal of **852 trials**, significantly reducing the dataset size.
    - A more lenient **40% threshold** removed **468 trials**. This was identified as a potentially better balance, as it discarded the most corrupted trials while retaining a larger portion of the data for training.

### Experiment 2: Multi-Stage Artifact Removal Pipelines
- **Source Notebooks**: `dropping-with-technique-1.ipynb`, `dropping-with-technique-2.ipynb`, `dropping-with-technique3.ipynb`
- **Goal**: To investigate the effectiveness of more complex, multi-stage pipelines for a more nuanced approach to artifact removal.
- **Methods Explored**:
    - **Technique 1**: A sequential process that first removed trials with motion artifacts, then applied per-subject standardization and Independent Component Analysis (ICA), and finally used a Z-score threshold to reject remaining outlier trials.
    - **Technique 2**: A similar pipeline that started with motion artifact rejection, followed by subject-wise data centering, ICA, and a final rejection stage based on Z-scores.
    - **Technique 3**: A combined approach that rejected trials based on both motion and Z-score thresholds *before* applying ICA for further cleaning.
- **Findings**: These notebooks demonstrated that a phased approach can be highly effective. For example, `dropping-with-technique-1.ipynb` showed that **396 trials** could be removed based on motion artifacts alone. Each technique provided a different trade-off between data retention and data cleanliness, highlighting that the optimal pipeline may depend on the specific requirements of the downstream task.

## 2. Normalization Techniques

These experiments aimed to address the significant inter-subject and inter-session variability in EEG signal amplitudes, a common challenge that can hinder the performance of generalized models.

### Experiment: Subject-Wise vs. Trial-Wise Normalization
- **Source Notebooks**: `subject-wise-normalization.ipynb`, `notebook060b441271.ipynb`
- **Goal**: To compare the impact of different data scaling strategies on the signal distribution.
- **Methods Compared**:
    - **Subject-Wise Normalization**: The mean and standard deviation were calculated across *all* trials belonging to a single subject. These statistics were then used to apply Z-score normalization to each of that subject's trials.
    - **Per-Trial Normalization**: The mean and standard deviation were calculated for *each individual trial*. Each trial was then normalized using its own statistics, forcing it to have a mean of 0 and a standard deviation of 1.
- **Findings**:
    - **Subject-wise normalization** successfully reduced inter-subject differences while preserving the natural variability and relative amplitude differences *between* trials for a single subject.
    - **Per-trial normalization** was a more aggressive approach, making the amplitude distribution of all trials nearly identical. While this can be beneficial for models sensitive to absolute scaling, it risks removing potentially useful information contained in trial-to-trial amplitude variations.

## 3. Integrated Preprocessing Pipelines

Several notebooks combined various techniques into a single, comprehensive workflow to transform raw data into a clean, analysis-ready format.

- **Source Notebook**: `preprocessing_thresholds.ipynb` (and others)
- **Goal**: To construct a logical sequence of steps that systematically addresses different types of noise and data issues.
- **Common Steps in the Explored Pipelines**:
    1.  **Artifact Masking**: The initial step in all pipelines was to identify and flag bad data segments using a combination of methods, including clipping detection, motion artifact detection, and high voltage thresholds. This confirmed that **100% of raw trials** contained some form of noise.
    2.  **Interpolation**: To salvage trials with minor issues, small gaps of contaminated data were often interpolated using surrounding clean data points (e.g., via linear interpolation).
    3.  **Filtering**: Standard signal processing filters were applied, including a bandpass filter (e.g., 4-40Hz) to isolate relevant neural frequencies and notch filters to remove power line noise.
    4.  **ICA Cleaning**: For trials with significant motion or muscle artifacts, Independent Component Analysis (ICA) was often used to identify and mathematically remove the noise components from the signal.
    5.  **Final Trial Rejection**: As a last quality check, a threshold (like the 40% rule) was often applied to discard any trials that remained heavily contaminated after the cleaning steps.