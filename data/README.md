# Dataset Description

⚠️ **Important Notice**  
The dataset is owned by the **Military Technical College** and was shared for competition use only.  
It **cannot be shared publicly**.  
If you need access, please contact **competition@mtc.edu.eg** to request permission or join the first phase of the Kaggle competition to get the first version of the dataset (without the full test set).  

🔗 [Kaggle Competition Link](https://www.kaggle.com/competitions/mtcaic3)

---

## MTC-AIC3 BCI Competition Dataset

### Dataset Overview
- **EEG Data**: Recordings from 8 channels  
- **Sampling Rate**: 250 Hz  
- **Participants**: 45 male subjects, average age 20 years  
- **Tasks**: Two Brain-Computer Interface (BCI) paradigms  
  - **Motor Imagery (MI)**: Imagining left or right hand movement  
  - **Steady-State Visual Evoked Potential (SSVEP)**: Focusing on visual stimuli  

**Classes**  
- **MI**: 2 classes → Left, Right  
- **SSVEP**: 4 classes → Left (10 Hz), Right (13 Hz), Forward (7 Hz), Backward (8 Hz)  

**Trial Duration**  
- **MI**: 9 seconds per trial  
  - 3.5s: Marker (preparation)  
  - 4s: Stimulation (imagine movement)  
  - 1.5s: Rest  
- **SSVEP**: 7 seconds per trial  
  - 2s: Marker (preparation)  
  - 4s: Stimulation (focus on stimulus)  
  - 1s: Rest  

**Trials per Session**: 10  

---

### Directory Structure
The dataset is organized into two main task directories (`MI/` and `SSVEP/`) inside the `mtc-aic3_dataset` folder.  
Each task contains three splits: `train/`, `validation/`, and `test/`.

```text
mtc-aic3_dataset/
├── MI/
│   ├── train/         # 30 subjects (4800 trials total)
│   ├── validation/    # 5 subjects (100 trials total)
│   └── test/          # 10 subjects (200 trials total)
├── SSVEP/
│   ├── train/         # 30 subjects
│   ├── validation/    # 5 subjects
│   └── test/          # 10 subjects
├── train.csv
├── validation.csv
├── test.csv
└── sample_submission.csv
```

# Dataset Structure

## Directory Layout
- Each **subject directory** (e.g., `S1/`, `S2/`) contains **session directories** (`1/`, `2/`, etc.).
- Inside each session folder is an **EEGdata.csv** file holding **10 concatenated trials**.

---

## Data Files

### EEGdata.csv (per session)
Each session file contains:

- **Columns**
  - `Time`
  - 8 EEG channels: `FZ`, `C3`, `CZ`, `C4`, `PZ`, `PO7`, `OZ`, `PO8`
  - Motion sensors: `AccX`, `AccY`, `AccZ`, `Gyro1–3`
  - `Battery` level
  - `Counter`
  - `Validation` flag

- **Samples per trial**
  - **MI (Motor Imagery):** 2250 samples (9s × 250 Hz)
  - **SSVEP (Steady-State Visual Evoked Potential):** 1750 samples (7s × 250 Hz)

---

### train.csv (4800 rows)
- Columns: `id`, `subject_id`, `task`, `trial_session`, `trial`, `label`

### validation.csv (100 rows)
- Same structure as `train.csv`, but smaller.

### test.csv (200 rows)
- Same as `validation.csv`, but **without labels**.

### sample_submission.csv
Format template for competition submission:

id,label
4901,Left
4902,Right
...


---

## How to Access a Trial

1. Pick a row from `train.csv`, `validation.csv`, or `test.csv`.
2. Use `subject_id`, `task`, and `trial_session` to locate the correct **EEGdata.csv**.
**Example:**
mtc-aic3_dataset/MI/train/S1/1/EEGdata.csv

3. Use the `trial` column to slice the correct segment:

- **MI trials:**  
(trial-1) * 2250 : trial * 2250

- **SSVEP trials:**  
(trial-1) * 1750 : trial * 1750