# **SSVEP Classification Pipeline**

This repository contains the complete code and assets for the Steady-State Visual Evoked Potential (SSVEP) classification task, part of the MTC-AIC3 BCI Competition. The solution is divided into two main components: model training and inference.

## **Prerequisites**

This project requires the official MTC-AIC3 BCI Competition dataset. As mentioned in the main repository's README, the dataset is not included here. Please download it from the official competition page and place it in a location accessible by the notebooks.

## **Directory Structure**

The SSVEP folder is organized as follows to separate the training and inference workflows:

SSVEP/  
├── Training/  
│   ├── ssvep-training.ipynb     \# Notebook to train the model from scratch.  
│   ├── outlier\_list.csv                  \# List of outlier trials to be excluded.  
│   └── imitation/                \# Supplementary dataset.  
│  
├── Inference/  
│   ├── ssvep-inference.ipynb    \# Notebook to run predictions with the pre-trained model.  
│   └── ssvep\_checkpoint\_filtered.pkl              \# The pre-trained model and assets.  
│  
└── README.md                             \# This file.

## **1\. Training**

The Training folder contains everything needed to train the classification model from the ground up.

### **Contents:**

* **ssvep\_training.ipynb**: A Jupyter Notebook that walks through the entire process of data loading, preprocessing, feature engineering, and model training. It generates the ssvep\_checkpoint.pkl file used for inference.  
* **outlier\_list.csv**: A CSV file specifying trials that were identified as outliers and should be excluded from the training process to improve model robustness.  
* **imitation/**: A supplementary dataset used to augment the primary training data, helping the model generalize better.

### **Usage:**

To retrain the model, open and run the ssvep\_training.ipynb notebook.

**Important**: Before running, please ensure the paths at the top of the notebook correctly point to the location of the main competition dataset and the supplementary imitation/.

## **2\. Inference**

The Inference folder provides a streamlined way to make predictions on new EEG data using the pre-trained model.

### **Contents:**

* **ssvep\_inference.ipynb**: A lightweight Jupyter Notebook designed for prediction. It loads the checkpoint, processes a target EEGdata.csv file, and outputs the classification results for all 10 trials within it.  
* **ssvep\_checkpoint\_filterd.pkl**: A pickle file containing the complete, trained pipeline. This includes the ensemble model, the feature scaler, the label encoder, and all necessary assets like spatial filters and signal templates.

### **Usage:**

To run predictions:

1. Open the ssvep\_inference.ipynb notebook.  
2. In the final cell, update the EEG\_FILE\_TO\_TEST variable to the full path of the EEGdata.csv file you wish to analyze.  
3. Run all the cells in the notebook.