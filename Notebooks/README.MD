## **Dataset acquisition and audit**

Here in this notebook, we'll download, verify, and organize the EatSense dataset for BitePulse AI: 

* Mount Google Drive, confirm free space, and set a single data root.
* Download EatSense data using a fast, resumable method.
* Verify archives for integrity before extraction.
* Extract archives into organized folders, including any zip files packed inside other zips.
* Generate a quick inventory of folders, file counts, and sizes.
* Sanity check the setup.

**Folders Plan:** 

* `.../eatsense/zips` holds the original downloads.
* `.../eatsense/rgb` holds RGB videos.
* `.../eatsense/depth` holds depth maps.
* `.../eatsense/poses_true` holds true face pose csv files.
* `.../eatsense/poses_fake` holds fake face pose csv files.
* `.../eatsense/test holds` the official test set.
* `.../eatsense/all_misc` holds documents and readme files

## **Label prep and task definition**

In this notebook let's turn EatSense data into clean training targets for our BitePulse AI model. so we'll define what counts as an intake event; convert frame level labels into segments with start time and end time; and freeze person disjoint splits for train, validation, and test. By the end, we will have compact label tables that downstream notebooks can load directly.

We'll utilize the data we saved on our google drive: 

* Manifest file I created earlier
  * Path: /content/drive/MyDrive/eatsense/manifest_eatsense.csv
  * Purpose: join keys across modalities and keep everything aligned.

* 2D pose CSVs for training features and labels
  * Path: /content/drive/MyDrive/eatsense/poses_true

* RGB videos for quick visual checks only
  * Path: /content/drive/MyDrive/eatsense/rgb/deepfaked

I will not use depth in this notebook because my target app accepts user recorded RGB videos only. At inference time I can compute 2D poses from RGB, so training on 2D poses keeps training and deployment aligned.

