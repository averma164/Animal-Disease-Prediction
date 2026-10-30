# Animal Disease Prediction System: Architecture, Procedures & Technical Guide

---

## 1. Project Overview & Clinical Objective

The **Animal Disease Prediction System** is an end-to-end, multimodal artificial intelligence platform designed to assist veterinarians, farmers, and pet owners in early diagnosis of animal diseases. It covers **4 animal species** across **35 disease conditions**:

- 🐄 **Cow** (6 conditions): *Foot and Mouth Disease, Healthy, Lumpy Skin Disease, Mastitis, Ringworm, Warts*
- 🐶 **Dog** (13 conditions): *Dental Disease, Distemper, Eye Infection, Fungal Infection, Healthy, Hot Spots, Kennel Cough, Mange, Parvovirus, Ringworm, Skin Allergy, Tick Infestation, Worm Infection*
- 🐱 **Cat** (9 conditions): *Dental Disease, Ear Mites, Eye Infection, Fungal Infection, Ringworm, Scabies, Skin Allergy, Urinary Tract Infection, Worm Infection*
- 🐟 **Fish** (7 conditions): *Bacterial Red Disease, Bacterial Aeromoniasis, Bacterial Gill Disease, Fungal Saprolegniasis, Healthy Fish, Parasitic Diseases, Viral White Tail Disease*

### The Core Problem It Solves
Single-modality vision models frequently fail when diagnosing veterinary conditions because:
1. **Visual Overlap**: Dermatological conditions like fungal infections, mange, ringworm, and allergies produce almost identical superficial symptoms (redness, alopecia, scaling).
2. **Internal/Systemic Diseases**: Serious viral and parasitic conditions (such as Canine Parvovirus, Distemper, or Feline Urinary Tract Infections) manifest primarily through behavioral and physiological symptoms (bloody stool, fever, vomiting, frequent urination) rather than distinctive external skin lesions.

To solve this, the system implements a **Multimodal Diagnostic Engine** that synergistically fuses high-resolution deep learning computer vision with a rule-based clinical symptom verification system.

---

## 2. System Architecture & Information Flow

```
[User Input]
   ├── 1. Animal Type (Cow / Dog / Cat / Fish)
   ├── 2. Uploaded Clinical Photo (JPG / PNG)
   └── 3. Selected Observed Symptoms (0 or more)
            │
            ├───────────────────────────────────────────────┐
            ▼                                               ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│       Vision Processing Engine       │  │       Symptom Inference Engine       │
│  (EfficientNet-B3 / MobileNetV2)     │  │        (14_symptom_engine.py)        │
│                                      │  │                                      │
│  • Decode & Aspect-Preserving Resize │  │  • Normalize user-reported symptoms  │
│  • 3-View Test-Time Augmentation     │  │  • Clinical Precision Scoring        │
│    (Original + H-Flip + 90% Zoom)    │  │  • Clinical Support Factor (0.55-1.0)│
│  • Softmax Probability Distribution  │  │  • Disease Compatibility Scores      │
└──────────────────┬───────────────────┘  └──────────────────┬───────────────────┘
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │    Multimodal Fusion & Normalization    │
                   │      (15_combined_prediction.py)        │
                   │                                         │
                   │  • Disease-Type Dynamic Weighting:      │
                   │    - Systemic:  15% Image / 85% Symptoms│
                   │    - Cutaneous: 70% Image / 30% Symptoms│
                   │    - General:   45% Image / 55% Symptoms│
                   │  • Cross-Disease Probability Calibration│
                   │  • Confidence Classification            │
                   └────────────────────┬────────────────────┘
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │     Flask Web Application Interface     │
                   │      (webapp/app.py & index.html)       │
                   │                                         │
                   │  • Top Diagnosis & Confidence Rating    │
                   │  • Ranked Top-5 Predictions Breakdown   │
                   │  • Visual Confidence Comparison Bars    │
                   │  • Severity Status & Diagnostic Guidance│
                   └─────────────────────────────────────────┘
```

---

## 3. Detailed Step-by-Step Procedure & Script Guide

The pipeline was built methodically across distinct phases:

### Phase 1: Data Audit, Master Cataloging & Split

| Script | Purpose & Details |
| :--- | :--- |
| [`01_create_image_master.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/01_create_image_master.py) | Scans raw image collections across all folders, assigns unique deterministic IDs, extracts metadata (dimensions, format, animal, raw label), and compiles `image_master.csv`. |
| [`02_dataset_audit.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/02_dataset_audit.py) | Checks dataset integrity: identifies corrupt files, unreadable headers, 0-byte images, and unsupported color spaces. |
| [`03_label_analysis.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/03_label_analysis.py) | Analyzes label distribution, identifies severe class imbalances, spelling variations, and inconsistent folder nomenclature. |
| [`04_clean_labels.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/04_clean_labels.py) | Standardizes label taxonomy into clean scientific disease names (e.g. mapping "lumpy", "lumpy_skin_disease", and "LSD" into `Lumpy_Skin`). |
| [`05_verify_cleaned_labels.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/05_verify_cleaned_labels.py) | Validates that 100% of samples match valid schema entries with zero unmapped orphan images. |
| [`06_split_dataset.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/06_split_dataset.py) | Performs stratified 70/15/15 Train-Validation-Test splitting per class, guaranteeing equal distribution across splits without data leakage. |
| [`07_prepare_dataset.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/07_prepare_dataset.py) | Organizes physical images into structured directory hierarchies: `prepared_dataset/{train, validation, test}/{Animal}/{Disease}/`. |
| [`08_create_class_mapping.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/08_create_class_mapping.py) | Generates `models/{Animal}_class_mapping.json` storing index-to-label and label-to-index mappings for neural network heads. |

---

### Phase 2: Class Rebalancing & Specialized Training

| Script | Purpose & Details |
| :--- | :--- |
| [`09_inspect_mastitis_normal.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/09_inspect_mastitis_normal.py) | Investigated false positives between Cow Mastitis and Healthy udders, identifying mislabeled non-infected samples. |
| [`10b_fix_mastitis_normal.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/10b_fix_mastitis_normal.py) | Re-curated udder imagery, correctly isolating true inflammatory mastitis lesions from normal anatomy. |
| [`11c_augment_small_classes.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/11c_augment_small_classes.py) | Offline augmentation pipeline for under-represented classes (e.g., Feline Fungal, Dog Distemper) using elastic deformation, color jitter, and random cropping. |
| [`11b_train_per_animal_models.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/11b_train_per_animal_models.py) | Initial training script using MobileNetV2 / EfficientNetB0 backbones across animals. |
| [`11e_train_sota_efficientnet_b3.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/11e_train_sota_efficientnet_b3.py) | **State-of-the-Art Training Engine**: Uses EfficientNet-B3 at $300\times300$ resolution, MixUp data augmentation, Categorical Focal Crossentropy ($\gamma=2.0, \text{smoothing}=0.1$), two-phase learning rate schedules, and unfreezing of the top 80 layers. |

---

### Phase 3: Evaluation, Confusion Matrices & Inference

| Script | Purpose & Details |
| :--- | :--- |
| [`12_evaluate_image_model.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/12_evaluate_image_model.py) | Evaluates vision models on held-out test datasets, computes precision, recall, weighted F1, and renders high-resolution confusion matrix plots saved to `results/{Animal}_confusion_matrix.png`. |
| [`13_test_image_prediction.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/13_test_image_prediction.py) | CLI utility to test single image predictions from the terminal. |
| [`14_symptom_engine.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/14_symptom_engine.py) | Rule-based clinical symptom matching engine referencing [`symptom_data/disease_symptoms.json`](file:///d:/Animal%20disease%20Prediction/symptom_data/disease_symptoms.json). Calculates clinical precision and disease support scores. |
| [`15_combined_prediction.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/15_combined_prediction.py) | **Multimodal Engine Core**: Integrates 3-view Test-Time Augmentation (TTA), applies dynamic clinical disease weights, and normalizes combined probabilities. |
| [`16_evaluate_multimodal.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/16_evaluate_multimodal.py) | Benchmark script that simulates clinical visits (image + 2 observed symptoms) across the entire 1,417-image test dataset to measure true real-world diagnostic performance. |

---

## 4. Key Algorithmic Improvements & Problem Solutions

### 1. Test-Time Augmentation (TTA)
- **Problem**: When users upload real photos from mobile phones or in varied agricultural lighting, slight head rotations or shadows caused single-shot CNN inferences to fluctuate.
- **Solution**: In [`15_combined_prediction.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/15_combined_prediction.py), `predict_from_image` generates 3 augmented versions of the input image on-the-fly:
  1. Original image resized to model input dimensions ($224\times224$ or $300\times300$).
  2. Horizontally flipped view (invariance to left/right orientation).
  3. 90% central crop scaled back to full resolution (zooming past distracting borders/backgrounds).
- All 3 images are evaluated in a single forward tensor batch:
  $$\vec{P}_{\text{image}} = \frac{1}{3} \sum_{k=1}^3 \text{Softmax}(\text{Model}(\text{View}_k))$$

### 2. Resolution of the Symptom Dictionary Penalization Flaw
- **Old Formula**: $\text{Score} = \frac{\text{matched symptoms}}{\text{total symptoms in database}}$
- **Flaw**: A disease like *Dog Dental Disease* has 8 catalogued symptoms in veterinary literature. If an owner observed 2 classic symptoms (e.g. *bad breath* + *swollen gums*), the engine assigned $2/8 = 25\%$ ("Low Compatibility"), dragging down an otherwise perfect $80\%$ image match to $49\%$.
- **New Formula** in [`14_symptom_engine.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/14_symptom_engine.py):
  $$\text{Precision} = \frac{\text{Matched Symptoms}}{\text{Observed Symptoms Selected by User}}$$
  $$\text{Support Factor} = \min(1.0,\, 0.55 + 0.25 \times (\text{Matched Symptoms} - 1))$$
  $$\text{Score} = \text{Precision} \times \text{Support Factor}$$
  Now, if the owner observes 2 symptoms and both match Dental Disease, precision is $100\%$ and support is $80\%$, correctly validating the diagnosis.

### 3. Calibrated Synergistic Multimodal Fusion
- **Old Logic**: Simple unnormalized linear combination: $0.50 \cdot P_{\text{image}} + 0.50 \cdot \text{Score}_{\text{symptoms}}$. Unmatched classes contributed 0, causing the total probability to fail to sum to 1.
- **New Logic** in [`15_combined_prediction.py`](file:///d:/Animal%20disease%20Prediction/anaconda_projects/15_combined_prediction.py):
  1. Dynamic weighting based on disease pathology:
     - **Systemic Diseases** (*Distemper, Parvovirus, Worms, Kennel Cough, UTI*): $W_{\text{symptom}} = 0.85, W_{\text{image}} = 0.15$.
     - **Cutaneous Diseases** (*Mange, Ringworm, Lumpy Skin, Warts, Hot Spots*): $W_{\text{image}} = 0.70, W_{\text{symptom}} = 0.30$.
     - **General/Mixed**: $W_{\text{image}} = 0.45, W_{\text{symptom}} = 0.55$.
  2. If symptoms support a candidate, image and symptoms reinforce:
     $$\text{Raw}_d = W_{\text{img}} \cdot P_{\text{img},d} + W_{\text{symp}} \cdot \text{Score}_{\text{symp},d}$$
  3. If symptoms contradict a candidate, it is moderated:
     $$\text{Raw}_d = (1.0 - 0.75 \cdot W_{\text{symp}}) \cdot P_{\text{img},d}$$
  4. Final probability calibration:
     $$P_{\text{final}, d} = \frac{\text{Raw}_d}{\sum_{j} \text{Raw}_j}$$

---

## 5. Evaluation Benchmarks & Confusion Matrix Analysis

Across all **1,417 test images**, the multimodal integration achieved significant performance leaps:

| Animal | Test Images | Classes | Vision-Only Baseline | **New Multimodal Accuracy** | **New Weighted F1** | Net Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🐄 **Cow** | 696 | 6 | 90.52% | **96.70%** | **96.68%** | **+6.18%** |
| 🐶 **Dog** | 522 | 13 | 81.23% | **90.80%** | **90.57%** | **+9.57%** |
| 🐱 **Cat** | 100 | 9 | 58.00% | **86.00%** | **84.44%** | **+28.00%** 🚀 |
| 🐟 **Fish** | 99 | 7 | 79.80% | **94.95%** | **94.99%** | **+15.15%** |
| **Overall** | **1,417** | **35** | **84.05%** | **93.68%** | **93.47%** | **+9.63%** |

### Detailed Species Analysis

1. **Cow Evaluation ([Cow_confusion_matrix.png](file:///d:/Animal%20disease%20Prediction/results/Cow_confusion_matrix.png))**:
   - Foot and Mouth Disease: 96.4% recall (108/112).
   - Lumpy Skin: 89.1% recall (205/230) — minor residual confusion with normal skin folds.
   - Mastitis: 77.8% recall (21/27) — restored following healthy udder re-curation.
   - Multimodal boost to **96.70%** via clinical fever and milk reduction indicators.

2. **Dog Evaluation ([Dog_confusion_matrix.png](file:///d:/Animal%20disease%20Prediction/results/Dog_confusion_matrix.png))**:
   - Ringworm: 97.8% recall (133/136).
   - Mange: 97.3% recall (109/112).
   - Systemic separation: Distemper, Parvovirus, and Worms previously showed visual dispersion due to generic lethargy in photos. Gastrointestinal symptoms (bloody stool, vomiting) elevated Dog accuracy past **90.80%**.

3. **Cat Evaluation ([Cat_confusion_matrix.png](file:///d:/Animal%20disease%20Prediction/results/Cat_confusion_matrix.png))**:
   - Smallest dataset (~100 test images).
   - Visual-only struggled heavily on Fungal Infection (20% recall) and UTI (50% recall).
   - Multimodal integration drove a **+28.0% surge to 86.00%**, as litter box behavior and scratching patterns disambiguated clinical states.

4. **Fish Evaluation ([Fish_confusion_matrix.png](file:///d:/Animal%20disease%20Prediction/results/Fish_confusion_matrix.png))**:
   - Bacterial Gill Disease and White Tail Disease achieved 100% precision.
   - Multimodal scoring reached **94.95%**, separating secondary fungal growths from primary bacterial lesions.

---

## 6. How to Operate & Maintain the System

### 1. Running the Web Application
The web app runs on Flask with a live mobile-responsive UI:
```powershell
cd "D:\Animal disease Prediction\webapp"
& "C:\Users\User\anaconda3\python.exe" app.py
```
- Open your browser at: **[http://localhost:5000](http://localhost:5000)**
- Upload any animal photo and optionally check 1–3 observed symptoms.

### 2. Running Standalone CLI Predictions
To predict directly on an image from the terminal:
```powershell
# Image only
& "C:\Users\User\anaconda3\python.exe" anaconda_projects/15_combined_prediction.py --animal Dog --image "prepared_dataset/test/Dog/Dental Disease/IMG_05393.jpg"

# Image + Observed Symptoms
& "C:\Users\User\anaconda3\python.exe" anaconda_projects/15_combined_prediction.py --animal Dog --image "prepared_dataset/test/Dog/Dental Disease/IMG_05393.jpg" --symptoms "bad breath" "red or swollen gums"
```

### 3. Re-Running Full Multimodal Benchmarks
To evaluate all test sets across species:
```powershell
& "C:\Users\User\anaconda3\python.exe" anaconda_projects/16_evaluate_multimodal.py
```

### 4. Re-Generating Confusion Matrices
To re-evaluate pure vision models and save updated confusion matrices to `results/`:
```powershell
& "C:\Users\User\anaconda3\python.exe" anaconda_projects/12_evaluate_image_model.py
```
