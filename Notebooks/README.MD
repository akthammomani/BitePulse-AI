## **Dataset acquisition and audit**

Here in this notebook, we'll download, verify, and organize the EatSense dataset for BitePulse AI: 

* Mount Google Drive, confirm free space, and set a single data root.
* Download EatSense data using a fast, resumable method.
* Verify archives for integrity before extraction.
* Extract archives into organized folders, including any zip files packed inside other zips.
* Generate a quick inventory of folders, file counts, and sizes.
* Sanity check the setup.

## **Label prep and task definition**

In this notebook, we'll turn EatSense data into clean training targets for our BitePulse AI model. so we'll define what counts as an intake event; convert frame level labels into segments with start time and end time; and freeze person disjoint splits for train, validation, and test. By the end, we will have compact label tables that downstream notebooks can load directly.

We'll utilize the data we saved on our google drive: 

* Manifest file I created earlier
  * Path: /content/drive/MyDrive/eatsense/manifest_eatsense.csv
  * Purpose: join keys across modalities and keep everything aligned.

* 2D pose CSVs for training features and labels
  * Path: /content/drive/MyDrive/eatsense/poses_true

* RGB videos for quick visual checks only
  * Path: /content/drive/MyDrive/eatsense/rgb/deepfaked

I will not use depth in this notebook because my target app accepts user recorded RGB videos only. At inference time I can compute 2D poses from RGB, so training on 2D poses keeps training and deployment aligned.

## **Feature Pipeline**

Our goal in this notebook is to convert labeled meals into model-ready inputs by:

* Cutting videos/poses into fixed windows,
* Extracting pose & motion features (hand&rarr;mouth distance, wrist speed, elbow angle, etc.), and
* Saving tensors + labels to disk for fast training.
We'll also add a minimal metrics scaffold (precision/recall for intake events) to quickly sanity-check feature quality.

We'll utilize the data we saved on our google drive from Label prep and task definition notebook - (label_v1):

**Our inputs** from labels_v1: 

* manifest_with_split.parquet: video paths + split
* frames_idx.parquet: per-frame labels (label, time_sec, key, split)
* segs_idx.parquet: intake segments (start_sec, end_sec, label, key, split)
* subject_split.parquet: to avoid subject leakage 
* true2d_parquet/<key>.parquet: per-frame 2D joints 

## Model Comparison

BitePulse AI went through four modeling stages:

1. **Baseline TCN (Pose-only)**: Initial window-based TCN over pose features.  
   - Good ROC AUC but very low PR AUC and F1, meaning it mostly learned “non-intake” and missed many true bites.

2. **Hyperband-tuned TCN**: Same architecture with Keras Tuner.  
   - Slight ROC/PR AUC improvements, but still struggled with recall and precision on rare INTAKE windows.

3. **RGB 3D-CNN**: Heavy video model trained on RGB frame stacks.  
   - Training was very expensive and PR AUC actually dropped, showing that RGB alone wasn’t enough under our compute budget.

4. **Final Model: MS-TCN (Frame-Level)**: Multi-stage TCN trained on frame-level pose sequences.  
   - **Best overall metrics:** macro Precision **0.595**, Recall **0.887**, F1 **0.614**, ROC AUC **0.955**, PR AUC **0.605** (see results table below).  
   - Delivers high recall for INTAKE while keeping precision reasonable, which is critical for bite detection.

> Table: Model metrics across all experiments (Precision, Recall, F1, ROC AUC, PR AUC).

![Image](https://github.com/user-attachments/assets/be9cf6bc-1813-4b17-98c9-9012f40c5f96)

### Precision–Recall

The PR curves below show how the final MS-TCN sharply outperforms earlier models on PR AUC, reflecting much better performance on the rare INTAKE class.

![Image](https://github.com/user-attachments/assets/2529d385-c333-4ece-bf0d-3141f2e69177)

### ROC Curves

All models achieve strong ROC AUC, but MS-TCN achieves the highest area, confirming robust separation between INTAKE and NON-INTAKE frames.

![Image](https://github.com/user-attachments/assets/0489a075-dbc2-4821-ac4f-62166c93ee75)

### Confusion Matrices

The confusion matrices highlight the same story:  
MS-TCN dramatically increases true positives for INTAKE while keeping false positives at a manageable level, compared to the earlier TCN and RGB 3D-CNN baselines.

![Image](https://github.com/user-attachments/assets/f67d03e8-fd8a-4cca-9b97-358dcc3ae362)






