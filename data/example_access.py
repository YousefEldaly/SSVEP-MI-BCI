import pandas as pd
import os

base_path = './mtc-aic3_dataset/'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

def load_trial_data(row, base_path='.'):
    # Determine dataset split
    if row['id'] <= 4800:
        dataset = 'train'
    elif row['id'] <= 4900:
        dataset = 'validation'
    else:
        dataset = 'test'

    # Path to EEG data
    eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
    eeg_data = pd.read_csv(eeg_path)

    # Trial slicing
    samples_per_trial = 2250 if row['task'] == 'MI' else 1750
    start_idx = (int(row['trial']) - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial

    return eeg_data.iloc[start_idx:end_idx]

# Example usage
first_trial = train_df.iloc[0]
trial_data = load_trial_data(first_trial, base_path=base_path)
print(trial_data.head())
