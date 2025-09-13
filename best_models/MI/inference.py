BASE_PATH = "mtcaic3"
BEST_MODEL_PATH = "MI/checkpoint/best_mi_model.h5"
import pandas as pd
import numpy as np
import os
import mne
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm



warnings.filterwarnings('ignore')


# Define subjects and sessions to drop
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

def main():
    print("Step 1: Loading and filtering the training data index...")
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))

    mi_train_df = train_df[train_df['task'] == 'MI'].copy()

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



    print("\nStep 2: Calculating csp wieghts from training data as done during trainingx...")
    X_train = []
    y_train_labels = []
    for i, row in tqdm(mi_train_df.iterrows(), total=len(mi_train_df), desc="Processing Train Data"):
        X_train.append(load_and_preprocess_trial(row, BASE_PATH, is_train=True))
        y_train_labels.append(row['label'])

    X_val = []
    y_val_labels = []


    X_train = np.array(X_train)
    X_val = np.array(X_val)


    print("\nStep 3: Encoding labels...")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    print(f"Label mapping: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")


    print("\nStep 4: Applying Common Spatial Patterns (CSP)...")

    csp = mne.decoding.CSP(
        n_components=N_CSP_COMPONENTS, reg=None, log=None, transform_into='csp_space'
    )

    csp.fit(X_train, y_train)



    print("--- Starting Combined Inference Pipeline for MI Task ---")

    print("\nStep 1: Loading and splitting the test data by task...")
    try:
        test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
        
        mi_test_df = test_df[test_df['task'] == 'MI'].reset_index(drop=True)

        print(f"Found {len(mi_test_df)} MI trials")
    except FileNotFoundError:
        print(f"Error: 'test.csv' not found at '{os.path.join(BASE_PATH, 'test.csv')}'")



    print("\n--- PART B: Generating MI Predictions ---")

    print("Loading the saved MI model...")
    try:
        mi_label_encoder = label_encoder 
        mi_model = tf.keras.models.load_model(BEST_MODEL_PATH)
        print("Model 'best_mi_model.h5' loaded successfully.")
    except Exception as e:
        print(f"Error loading MI model or artifacts: {e}")
        print("Please ensure the MI training cell was run to generate the model and artifacts.")

    print("Preprocessing MI test data...")
    X_test_mi = []
    for i, row in tqdm(mi_test_df.iterrows(), total=len(mi_test_df), desc="Processing MI Trials"):
        X_test_mi.append(load_and_preprocess_trial(row, BASE_PATH, is_train=False))

    X_test_mi = np.array(X_test_mi)

    X_test_csp = csp.transform(X_test_mi)

    X_test_cnn = X_test_csp.transpose(0, 2, 1)
    print(f"Shape of MI test data for CNN: {X_test_cnn.shape}")

    y_test_probs = mi_model.predict(X_test_cnn)
    y_test_pred_indices = (y_test_probs > 0.51).astype(int).flatten()
    mi_final_labels = mi_label_encoder.inverse_transform(y_test_pred_indices)

    mi_results_df = pd.DataFrame({'id': mi_test_df['id'], 'label': mi_final_labels})
    print("MI predictions generated.")
    mi_results_df.to_csv('MI/inference_results/mi_predictions.csv', index=False)

if __name__ == "__main__":
    main()
    print("Inference completed successfully. Results saved to 'MI/inference_results/mi_predictions.csv'.")