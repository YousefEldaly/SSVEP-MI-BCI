BASE_PATH = r"mtcaic3"



import pandas as pd
import numpy as np
import os
import joblib
import mne
import random # Import the random library
import tensorflow as tf # Import tensorflow
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm

# --- 1. Setup, Constants, and Seeding ---

SAVE_DIR = "MI/reproduced_model_and_results/"
os.makedirs(SAVE_DIR, exist_ok=True) # Ensure directory exists

# --- Set a Random Seed for Reproducibility ---
SEED = 1
print(f"Using random seed: {SEED}")
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DROPS = {
    'S2': [7], 'S3': [7], 'S5': [6, 7], 'S6': [3], 'S8': [1, 2, 3, 4, 5],
    'S9': [1, 5, 7], 'S10': [1], 'S11': [2, 4, 7, 8], 'S12': [3, 4, 5, 6, 7],
    'S13': [1, 2, 3, 4, 7], 'S14': [1], 'S15': [7], 'S17': [6],
    'S18': [1, 2, 3, 4, 6, 7, 8], 'S19': [1, 2, 3, 4, 5, 6, 7, 8],
    'S21': [1, 2, 3, 4, 6, 8], 'S22': [1, 6], 'S23': [1, 2, 3], 'S24': [1, 3],
    'S25': [1], 'S27': [1, 4, 5], 'S28': [8], 'S29': [4, 7], 'S30': [1, 3]
}

MI_SAMPLES_PER_TRIAL = 2250
CHANNELS_TO_USE = ['C3', 'CZ', 'C4']
N_CSP_COMPONENTS = 2
AUGMENTATION_STRENGTH = 0.05


# --- 2. Data Filtering ---

print("\nStep 1: Loading and filtering the training data index...")
train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
validation_df = pd.read_csv(os.path.join(BASE_PATH, 'validation.csv'))

mi_train_df = train_df[train_df['task'] == 'MI'].copy()
mi_validation_df = validation_df[validation_df['task'] == 'MI'].copy()

indices_to_drop = []
for index, row in mi_train_df.iterrows():
    subject = row['subject_id']
    session = row['trial_session']
    if subject in DROPS and session in DROPS[subject]:
        indices_to_drop.append(index)

print(f"Original MI training trials: {len(mi_train_df)}")
mi_train_df.drop(indices_to_drop, inplace=True)
print(f"MI training trials after dropping specified sessions: {len(mi_train_df)}")
mi_train_df.reset_index(drop=True, inplace=True)

print("\nClass balance in the filtered training data:")
print(mi_train_df['label'].value_counts())


# --- 3. Data Loading & Preprocessing Function ---

def load_and_preprocess_trial(row, base_path, is_train=False):
    id_num = row['id']
    if id_num <= 4800: dataset_split = 'train'
    elif id_num <= 4900: dataset_split = 'validation'
    else: dataset_split = 'test'
    
    path = os.path.join(
        base_path, row['task'], dataset_split,
        row['subject_id'], str(row['trial_session']), 'EEGdata.csv'
    )
    
    eeg_data = pd.read_csv(path)
    
    start_idx = (row['trial'] - 1) * MI_SAMPLES_PER_TRIAL
    end_idx = start_idx + MI_SAMPLES_PER_TRIAL
    trial_data = eeg_data.iloc[start_idx:end_idx]

    trial_data_channels = trial_data[CHANNELS_TO_USE].values

    if is_train:
        noise = np.random.normal(0, AUGMENTATION_STRENGTH, trial_data_channels.shape)
        trial_data_channels += noise
        
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(trial_data_channels)
    
    return normalized_data.T


# --- 4. Prepare Datasets ---

print("\nStep 2: Preparing training and validation datasets...")
X_train = []
y_train_labels = []
for i, row in tqdm(mi_train_df.iterrows(), total=len(mi_train_df), desc="Processing Train Data"):
    X_train.append(load_and_preprocess_trial(row, BASE_PATH, is_train=True))
    y_train_labels.append(row['label'])

X_val = []
y_val_labels = []
for i, row in tqdm(mi_validation_df.iterrows(), total=len(mi_validation_df), desc="Processing Validation Data"):
    X_val.append(load_and_preprocess_trial(row, BASE_PATH, is_train=False))
    y_val_labels.append(row['label'])

X_train = np.array(X_train)
X_val = np.array(X_val)


# --- 5. Label Encoding ---

print("\nStep 3: Encoding labels...")
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_labels)
y_val = label_encoder.transform(y_val_labels)
print(f"Label mapping: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")


# --- 6. CSP Feature Extraction ---

print("\nStep 4: Applying Common Spatial Patterns (CSP)...")
csp = mne.decoding.CSP(
    n_components=N_CSP_COMPONENTS, reg=None, log=None, transform_into='csp_space'
)

csp.fit(X_train, y_train)

X_train_csp = csp.transform(X_train)
X_val_csp = csp.transform(X_val)

X_train_cnn = X_train_csp.transpose(0, 2, 1)
X_val_cnn = X_val_csp.transpose(0, 2, 1)

print(f"Shape of training data for CNN: {X_train_cnn.shape}")
print(f"Shape of validation data for CNN: {X_val_cnn.shape}")


# --- 7. CNN Model Definition ---

print("\nStep 5: Defining the CNN model...")
input_shape = (MI_SAMPLES_PER_TRIAL, N_CSP_COMPONENTS)

model = Sequential([
    Input(shape=input_shape),
    Conv1D(filters=32, kernel_size=10, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=3),
    Conv1D(filters=64, kernel_size=10, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=3),
    Conv1D(filters=128, kernel_size=10, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=3),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- 8. Model Training ---

print("\nStep 6: Training the model...")
model_checkpoint = ModelCheckpoint(os.path.join(SAVE_DIR, 'reproduced_model_mi.h5'), save_best_only=True, monitor='val_accuracy', mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train_cnn,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_cnn, y_val),
    callbacks=[model_checkpoint, early_stopping]
)

print("\nTraining complete.")
if 'val_accuracy' in history.history and len(history.history['val_accuracy']) > 0:
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
else:
    print("Could not determine validation accuracy.")