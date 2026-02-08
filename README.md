<p align="center">
  <!-- Language / Core -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Plots-Matplotlib%20%7C%20Plotly-11557C?logo=python&logoColor=white">
  <img alt="OpenCV/FFmpeg" src="https://img.shields.io/badge/Video-OpenCV%20%7C%20FFmpeg-5C2D91?logo=opencv&logoColor=white">
  <img alt="MediaPipe" src="https://img.shields.io/badge/Landmarks-MediaPipe%20Pose%20%7C%20FaceMesh-34A853?logo=google&logoColor=white">

  <!-- Modeling / ML -->
  <img alt="MS-TCN" src="https://img.shields.io/badge/Model-MS--TCN%20(frame--level)-0EA5E9">
  <img alt="Baselines" src="https://img.shields.io/badge/Baselines-Pose%20TCN%20%7C%203D%20CNN-F97316">
  <img alt="Pose features" src="https://img.shields.io/badge/Features-2D%20pose%20%7C%20wrist%20kinematics-10B981">
  <img alt="Imbalance" src="https://img.shields.io/badge/Loss-Class--weighted%20CE-8B5CF6">
  <img alt="Evaluation" src="https://img.shields.io/badge/Eval-ROC%20AUC%20%7C%20PR%20AUC-6366F1">
  <img alt="Privacy" src="https://img.shields.io/badge/Privacy-Pose--first%20%7C%20on--device--ready-F59E0B">

  <!-- Demo / App -->
  <img alt="Colab" src="https://img.shields.io/badge/Train-Google%20Colab-0F9D58?logo=googlecolab&logoColor=white">
  <img alt="Streamlit app" src="https://img.shields.io/badge/App-Streamlit%20%7C%20streamlit--webrtc-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Real time" src="https://img.shields.io/badge/Mode-Real--time%20bite%20detection-brightgreen">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-EatSense-4B5563">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-000000">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20prototype-22C55E">
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/82d8e765-a391-4110-b192-104216c7a644" 
       alt="Strivio logo" width="400" height="400" />
</p>



# **BitePulse AI: Real-Time Eating-Pace Feedback from Meal video** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://bitepulse-ai-test-1.streamlit.app//)
**Temporal Deep Learning for Bite Detection**


This project is the Capstone Project (AAI-590) for [the M.S. in Applied Artificial Intelligence](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Completed**

## **Table of contents**

- [Abstract](#abstract)
- [Introduction](#introduction)
- [BitePulse AI Audience](#bitepulse-ai-audience)
- [Data Summary](#data-summary)
- [Methods Used](#methods-used)
- [Technologies](#technologies)
- [Model Results](#model-results)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## **Abstract**

Most of us do not know how fast we really eat until a doctor, a coach, or our stomach complains. BitePulse AI asks whether a short phone video can provide that feedback in real time, without human scoring or sharing raw data. We train a series of temporal deep learning models on an annotated meal dataset to detect intake events (bites) and estimate eating pace, starting from a pose-based TCN and RGB 3D-CNN baselines and then moving to a frame-level MS-TCN over MediaPipe pose sequences, which clearly gives the best precision, recall, and PR AUC for the rare INTAKE class. For the live Streamlit demo, we use a lighter MediaPipe-based intake detector for latency and deployment reasons, and treat the MS-TCN as an offline gold-standard model that guides the design and target behavior of a future on-device pace coach.

## **Introduction**

Eating rate is an important but overlooked risk factor. Fast eating is linked to higher energy intake, weaker fullness, and digestive and metabolic problems, yet most people get no feedback about how fast they eat in daily life.

Researchers already measure eating pace by labeling “intake events” such as bites and sips in video and computing metrics like bites per minute. These methods work, but they depend on manual annotation and lab tools, not something you can put in a simple app.

BitePulse AI asks whether short meal videos, captured on a phone or laptop, can be turned into automatic bite detections and an easy-to-read pace summary in under one minute, without storing raw video. The project uses the EatSense dataset of anonymized meal videos with frame-level labels for eating, drinking, chewing, and resting to train supervised models for INTAKE vs NON_INTAKE and to build event-level bite timelines.

On the modeling side, BitePulse AI starts with pose-based TCN baselines and a compact RGB 3D-CNN, then moves to a frame-level Multi-Stage TCN (MS-TCN) over MediaPipe pose sequences, which gives the best precision, recall, and PR AUC for the rare intake class. For a live experience, a Streamlit app runs in the browser and uses MediaPipe landmarks with a lightweight wrist-to-mouth heuristic for real-time feedback, while the MS-TCN serves as an offline reference model. Together, these pieces support the core goal: practical, privacy-friendly eating-pace feedback built on temporal deep learning.


## **BitePulse AI Audience**

BitePulse AI serves both consumers (B2C) and wellness partners (B2B):
* For consumers (B2C), it helps everyday eaters who want better comfort after meals, calmer digestion, and more mindful control. A user can record a short meal clip and get clear feedback on their eating pace.
* For wellness partners (B2B), like nutrition coaches and digital health platforms, BitePulse AI delivers objective eating behavior signals (for example, slow pauses versus nonstop bursts) through an API, without anyone having to watch the video. In both cases, the goal is awareness and coaching, not medical diagnosis.

## **Data Summary**

The BitePulse AI prototype is built on the [EatSense dataset](https://groups.inf.ed.ac.uk/vision/DATASETS/EATSENSE/), a public collection of 135 real-world meal videos with anonymized faces and frame-level labels for actions such as eating, drinking, chewing, and resting, totaling about 14 hours of footage (Raza et al., 2023). For this project, these labels are collapsed into INTAKE (eat it) versus NON_INTAKE (all other actions).

We work with two main representations. For the window-based baselines, each meal is turned into overlapping fixed-length windows (for example, 0.5 seconds). Every window carries a short sequence of engineered pose features plus an index into the original RGB frames, and it is labeled positive if it overlaps an eating or drinking segment beyond a threshold. This produces a single, uniform table of training examples that can be shared across pose-only and RGB models.

For the MS-TCN experiments, we move directly to the frame level. The 16 EatSense actions are collapsed into binary frame labels, giving one long sequence of pose features and INTAKE/NON_INTAKE targets per session. Only about 5% of frames are INTAKE (compared with roughly 0.4% positive windows earlier), so the data remain highly imbalanced but contain a richer positive signal. Four of the 135 videos contain no intake frames; we keep them as realistic “no-intake” sessions that teach the model to predict zero bites when appropriate. Sequences are padded with an ignore index so that variable-length meals can be batched without losing context.

## **Methods Used**

- Frame-level labeling on EatSense: 16 original actions collapsed into **INTAKE vs NON-INTAKE**  
- Pose extraction with **MediaPipe Pose / FaceMesh** and temporal sequences of landmarks per video  
- **MS-TCN (Multi-Stage Temporal ConvNet)** over pose sequences for frame-level intake detection  
- Class-imbalance handling with **class-weighted cross-entropy** and careful split design (including 0-intake videos)  
- Temporal post-processing to turn frame scores into **bite events** (run-length filtering, merging, pacing metrics like BPM, intake %, IBI, pauses)  
- Model comparison across baselines (Pose TCN, Hyperband TCN, RGB 3D-CNN) using macro P/R/F1, ROC AUC, and PR AUC  

## **Technologies**

- Python 3 (Google Colab + local dev)
- **PyTorch** for MS-TCN training/inference, NumPy & pandas for data handling.
- **MediaPipe** (Pose & FaceMesh) as pre-trained landmark extractors. 
- **OpenCV / FFmpeg** for video -> frames processing.
- **Streamlit + streamlit-webrtc** for the BitePulse AI web app and real-time webcam ingestion.  
- Matplotlib & Plotly for evaluation plots and in-app visualizations. 
- Pre-trained **BitePulse MediaPipe checkpoint** loaded in the Streamlit app for live intake detection.


## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thanks to Professor Anna Marbut for guidance and feedback throughout the BitePulse AI capstone.

This project uses the **EatSense** dataset released by the University of Edinburgh School of Informatics.  
Raza, M. A., Chen, L., Nanbo, L., & Fisher, R. B. (2023). *EatSense: Human centric, action recognition and localization dataset for understanding eating behaviors and quality of motion assessment, Image and Vision Computing*.  
Dataset page: https://groups.inf.ed.ac.uk/vision/DATASETS/EATSENSE/
