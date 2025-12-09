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

Most of us have no idea how fast we really eat until a doctor, a coach, or a bad stomach reminds us. BitePulse AI asks a simple question: can a short phone video give people that feedback in real time, without human scoring or sharing their data? To explore this idea, we train a sequence of temporal deep learning models on a labeled meal dataset to detect intake events (bites) and estimate eating pace. We start with a pose-based Temporal Convolutional Network (TCN) as a lightweight baseline, then apply Hyperband tuning to the same architecture, and also train an RGB 3D-CNN on short frame clips to inject appearance cues. Finally, we move to a frame-level Multi-Stage TCN (MS-TCN) over MediaPipe pose sequences, which clearly dominates all prior models in macro precision, recall, F1, ROC AUC, and especially PR AUC for the rare INTAKE class. However, to keep latency, memory, and deployment complexity within the constraints of a browser-based Streamlit demo, the current app uses a lighter MediaPipe intake detector, with the MS-TCN serving as the “gold standard” offline model that informs the design and target behavior of a future on-device pace coach.

## **Introduction**

Eating rate is an overlooked behavioral risk factor. Experimental and observational studies show that rapid eating is associated with higher energy intake, weaker subjective satiety, and adverse gastrointestinal symptoms. In controlled meal studies, asking participants to eat slowly reduces how much they eat when allowed to serve themselves freely and increases post-meal fullness ratings (Andrade, Greene, & Melanson, 2008). Faster ingestion, on the other hand, is linked to greater postprandial reflux and gastric distension in clinical cohorts (Su et al., 2000). Meta-analytic work also connects fast eating with higher odds of metabolic syndrome and overweight (Zhu et al., 2020). Despite this evidence, most people receive little or no feedback about how fast they eat in everyday settings.
Researchers already treat eating pace as a measurable signal. In laboratory and free-living studies, teams annotate “intake events” such as bites on video, then compute metrics like bites per minute or burst patterns to study self-control, comfort, and energy intake (Rouast, Heydarian, Adam, & Rollo, 2020). However, these analyses typically depend on manual labeling and are not accessible to consumers or digital-health partners. There is a gap between what research systems can measure about eating behavior and the kind of timely, privacy-preserving feedback that could help individuals make small but meaningful changes in daily life.
BitePulse AI investigates whether modern temporal deep learning can close part of this gap. The project explores if short meal videos, captured on a phone, can be converted automatically into bite detections and an interpretable eating-pace summary that is returned to the user in under one minute. The primary end users are individual consumers who might benefit from gentle, real-time coaching about pace, and wellness organizations who may want objective but non-intrusive indicators of eating behavior for their programs. For consumers, the envisioned experience is a simple application that accepts a brief clip and returns a pace score, a timeline of detected intake events, and one sentence of neutral guidance. For partners, the same signals can be exposed through an API without exposing or storing identifiable video.
To support this investigation, we use the EatSense dataset, a public collection of real-world meal videos with anonymized faces and frame-level labels for eating, chewing, drinking, and resting (Raza et al., 2023). These annotations enable supervised learning of frame-level “intake versus non-intake” predictions and aggregation into event-level bite detections. In a deployed system, analogous data would come from user-recorded clips, processed either entirely on device or in a secure backend that discards raw video after inference.
The technical approach is to construct time-aligned sequences from the annotated segments and train temporal models that map pose-based features to binary intake labels. A pose-based Temporal Convolutional Network (TCN) operating on windowed features serves as a lightweight baseline that is compatible with on-device inference. We then apply Hyperband hyperparameter tuning to the same architecture and explore a compact 3D convolutional neural network (3D-CNN) over RGB clips to add appearance cues such as utensil contact and food visibility. Building on these baselines, we develop a frame-level Multi-Stage TCN (MS-TCN) over MediaPipe pose sequences, which substantially improves macro precision, recall, F1, ROC AUC, and especially PR AUC for the rare INTAKE class.
Finally, we prototype a live experience through a Streamlit web application that runs in the browser. For deployment constraints, the app uses MediaPipe-based landmark extraction with a lightweight intake heuristic for real-time feedback, while the MS-TCN serves as an offline “gold standard” model that informs the target behavior and metrics of the system. This demonstrates that the same pipeline can power both research-grade evaluation and a practical phone or web experience without persisting raw video.
The central hypothesis is that temporal models trained on labeled meal videos can achieve practically useful precision and recall for intake detection and produce stable, interpretable measures of eating pace that are suitable for real-time feedback. If this hypothesis is supported, BitePulse AI points toward a privacy-respecting, deployable “eating-pace coach” that brings methods currently used only in research labs into everyday life for consumers and wellness programs.


## **BitePulse AI Audience**

BitePulse AI serves both consumers (B2C) and wellness partners (B2B):
* For consumers (B2C), it helps everyday eaters who want better comfort after meals, calmer digestion, and more mindful control. A user can record a short meal clip and get clear feedback on their eating pace.
* For wellness partners (B2B), like nutrition coaches and digital health platforms, BitePulse AI delivers objective eating behavior signals (for example, slow pauses versus nonstop bursts) through an API, without anyone having to watch the video. In both cases, the goal is awareness and coaching, not medical diagnosis.

## **Data Summary**

The BitePulse AI prototype is built on the [EatSense dataset](https://groups.inf.ed.ac.uk/vision/DATASETS/EATSENSE/), a public collection of 135 real-world meal videos with anonymized faces and frame-level activity labels, totaling roughly 14 hours of footage and averaging about 11 minutes per clip (Raza et al., 2023). Each recording contains RGB video of a person eating at a table and a set of time-aligned annotations describing what the person is doing in each moment. For this project, the most important labels are eating, drinking, chewing, and resting, which we use to define intake versus non-intake behavior.

At a high level there are four layers of variables. At the session level, each meal has an identifier, basic context (for example, lunch versus snack), and camera setup. At the frame level, each image has a timestamp, a frame index, and pre-computed 2D body-pose key points for head, torso, arms, and hands. At the segment level, the dataset provides start and end times for labeled activities, such as a chewing phase or a drink. On top of this, our capstone introduces a window level: we slide a fixed-length window (for example, 0.5 seconds) over time with a fixed stride and assign each window a binary label indicating whether it contains an intake event.

The window representation is a key part of the novelty in our approach. Rather than train directly on variable-length segments, we create a uniform “grid” over time that can be used consistently across pose and RGB models. Each window carries (a) a short sequence of pose features, engineered from the 2D key points as relative positions to the head and simple velocities, and (b) an index into the original video frames so that the same window can be mapped to an RGB clip for the 3D-CNN. The target label for each window is derived from the segment annotations using an overlap rule: a window is positive if its time span overlaps an eating or drinking segment beyond a chosen threshold, and negative otherwise. This produces a single table of training examples that sits between the raw dataset and our models and makes it straightforward to compare pose-only and RGB baselines.

For the MS-TCN experiments we move one step closer to the raw annotations and work directly at the frame level. The 16 original EatSense actions are collapsed into a binary target (INTAKE for “eat it” and NON_INTAKE for all other labels), yielding long sequences of frame-wise pose features and labels for each session. Only about 5% of frames are INTAKE, compared with roughly 0.4% positive windows in the earlier table, so the data remain highly imbalanced but provide a richer positive signal. Four of the 135 videos contain no intake frames at all; we retain these sessions as realistic “no-intake” examples, which force the model to correctly predict zero bites when appropriate. Variable-length sequences are padded with an ignore index so that batches can be formed without discarding context at the start or end of meals.

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
