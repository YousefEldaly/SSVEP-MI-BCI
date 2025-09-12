# **SSVEP Classification Model Evolution**

This repository documents the iterative development of a machine learning pipeline for classifying Steady-State Visually Evoked Potential (SSVEP) signals from EEG data. The goal is to accurately determine the user's intended command (Forward, Backward, Left, Right) based on their brain activity.

The notebook ssvep-complete-incremental-testing.ipynb showcases the journey from a baseline model to a high-performing, state-of-the-art solution, with each major step detailed below.

## **Model Progression**

The model's accuracy was improved through a series of methodical enhancements in feature engineering, preprocessing, and classification techniques.

### **1\. Baseline: FBCCA (Weighted) \+ SVM**

* **Approach:** This initial model used a standard Filter-Bank Canonical Correlation Analysis (FBCCA) to extract features. The correlations from different frequency bands were weighted and summed to create a feature vector. A Support Vector Machine (SVM) with an RBF kernel was used for classification.  
* **Key Features:**  
  * Weighted FBCCA for feature extraction.  
  * SVM for classification.  
* **Performance:** \~46% Validation Accuracy.  
* **Analysis:** This baseline performed better than random chance (25%) but showed significant room for improvement. The confusion matrix revealed that the model struggled to distinguish between classes, particularly 'Left' and 'Right'.

### **2\. Concatenated Features \+ LDA**

* **Approach:** The first major improvement involved changing how features were constructed and classified. Instead of weighting and summing correlations, the correlation vectors from each filter bank were concatenated. This created a richer, more detailed feature vector. The classifier was switched to a Linear Discriminant Analysis (LDA) model, which is often effective for this type of data.  
* **Key Changes:**  
  * **Concatenated Features:** Preserved more information from the FBCCA process.  
  * **Chebyshev Filters:** Used more precise Chebyshev Type I filters.  
  * **LDA Classifier:** Replaced SVM with LDA, better suited for the feature space.  
* **Performance:** \~50% Validation Accuracy.

### **3\. Hybrid Features (TRCA) \+ Cross-Validation**

* **Approach:** This version introduced a powerful new feature extraction method, Task-Related Component Analysis (TRCA), which finds spatial filters that maximize the consistency of the signal for each class. These TRCA features were combined with the existing FBCCA features. To ensure the model was robust and generalized well, a **Leave-One-Subject-Out (LOSO)** cross-validation strategy was implemented.  
* **Key Changes:**  
  * **TRCA Features:** Added features based on signal reproducibility.  
  * **Hybrid Feature Set:** Combined TRCA and FBCCA features.  
  * **LOSO Cross-Validation:** A more rigorous validation method that trains on all subjects except one, which is held out for testing. This is repeated for every subject.  
  * **Classifier Exploration:** Tested multiple classifiers, with XGBoost showing strong performance.  
* **Performance:** Significant improvement, with XGBoost achieving the best results in this tier.

### **4\. Ensemble Modeling \+ PSD Features**

* **Approach:** To further boost performance, two key enhancements were made. A new feature type, **Power Spectral Density (PSD)**, was added to capture the signal power at target frequencies. More importantly, a **Voting Ensemble Classifier** was introduced, combining the strengths of XGBoost, SVC, and LDA.  
* **Key Changes:**  
  * **PSD Features:** Added signal power as a new feature dimension.  
  * **Voting Ensemble:** Combined predictions from three diverse models (XGBoost, SVC, LDA) using soft voting (based on predicted probabilities) for a more robust decision.  
* **Performance:** \~67% LOSO CV Accuracy.

### **5\. Optimal Time Window Selection**

* **Approach:** Analysis of the SSVEP trial structure revealed that the most valuable information was contained within a specific time window. The first two seconds of each trial were often preparatory, and the signal was strongest in the subsequent four seconds. The pipeline was adjusted to focus only on this optimal segment.  
* **Key Changes:**  
  * **Data Slicing:** Skipped the first 2 seconds and used only the next 4 seconds of each 7-second trial.  
* **Performance:** **\~71% LOSO CV Accuracy.** This was a critical breakthrough, demonstrating the importance of precise data selection.

### **6\. Final Model: Advanced Feature Engineering**

* **Approach:** This is the culmination of all previous learnings, incorporating several advanced feature engineering techniques to build the most powerful model.  
* **Key Changes:**  
  * **Common Average Reference (CAR):** A preprocessing step to reduce noise by subtracting the average signal across all channels.  
  * **Phase Locking Value (PLV):** Features that measure phase synchrony between different EEG channels.  
  * **Enhanced Harmonics:** Increased the number of harmonics from 3 to 5 to capture more signal information.  
  * **Subject-Specific Templates:** Created average signal templates for each subject and class, and used their correlation with the current trial as a powerful new feature.  
* **Performance:**  
  * **\~74% on Phase 1 Test Data.**  
  * **\~80% F1-Score on Phase 2 Test Data.**

## **How to Use**

1. **Environment:** Ensure you have a Python environment with the necessary libraries installed (pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, seaborn).  
2. **Data:** Place the competition data in the specified /kaggle/input/ directory structure.  
3. **Execution:** Run the cells in the ssvep-complete-incremental-testing.ipynb notebook sequentially. The final cells contain the code for training the best model and generating the submission.csv file.