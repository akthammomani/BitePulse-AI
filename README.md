<p align="center">
  <!-- Language / Core -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Torchvision" src="https://img.shields.io/badge/Torchvision-0.17%2B-FF6F00?logo=pytorch&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-plotting-11557C?logo=python&logoColor=white">
  <img alt="OpenCV/FFmpeg" src="https://img.shields.io/badge/Video-OpenCV%20%7C%20FFmpeg-5C2D91?logo=opencv&logoColor=white">

  <!-- Modeling / ML -->
  <img alt="TCN" src="https://img.shields.io/badge/Model-Temporal%20ConvNet-0EA5E9">
  <img alt="3D CNN" src="https://img.shields.io/badge/Model-3D%20CNN%20(RGB)-F97316">
  <img alt="Pose features" src="https://img.shields.io/badge/Features-2D%20pose%20%7C%20wrist%20kinematics-10B981">
  <img alt="Event-level eval" src="https://img.shields.io/badge/Evaluation-Event--level%20IoU-8B5CF6">
  <img alt="Real-time feedback" src="https://img.shields.io/badge/Goal-Real--time%20pace%20feedback-6366F1">
  <img alt="Privacy" src="https://img.shields.io/badge/Privacy-Pose--first%20%7C%20on--device%20ready-F59E0B">

  <!-- Demo / Ops -->
  <img alt="Colab" src="https://img.shields.io/badge/Train-Google%20Colab-0F9D58?logo=googlecolab&logoColor=white">
  <img alt="Streamlit demo" src="https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-EatSense-4B5563">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-000000">
  <img alt="Status" src="https://img.shields.io/badge/Status-Research%20prototype-brightgreen">
</p>

<p align="center">
  <img src="<!-- TODO: add BitePulse logo or representative image URL here -->"
       alt="BitePulse AI logo" width="330" height="330" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/64d9919e-ff96-4588-ae68-75c3f22b160e" 
       alt="Strivio logo" width="400" height="400" />
</p>

# **BitePulse AI: Real-Time Eating-Pace Feedback from Meal video** 
>Temporal Deep Learning for Bite and Sip Detection


This project is the Capstone Project (AAI-590) for [the M.S. in Applied Artificial Intelligence](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Ongoing**

## **Abstract**

Most of us have no idea how fast we really eat until a doctor, a coach, or a bad stomach reminds us. BitePulse AI asks a simple question: can a short phone video give people that feedback in real time, without human scoring or sharing their data? To explore this idea, we train temporal deep learning models on a labeled meal dataset to detect intake events (bites and sips) and estimate eating pace. A pose-based Temporal Convolutional Network is used as a lightweight baseline, and a 3D-CNN on RGB clips adds appearance cues. We report window-level and event-level performance, examine precision–recall tradeoffs, and discuss what these results imply for a future, privacy-respecting, on-device “pace coach” that responds in under a minute.

## **Introduction**

Eating rate is an overlooked behavioral risk factor. Experimental and observational studies show that rapid eating is associated with higher energy intake, weaker subjective satiety, and adverse gastrointestinal symptoms. In controlled meal studies, asking participants to eat slowly reduces ad libitum intake and increases post-meal fullness ratings. Faster ingestion, on the other hand, is linked to greater postprandial reflux and gastric distension in clinical cohorts. Meta-analytic work also connects fast eating with higher odds of metabolic syndrome and overweight. Despite this evidence, most people receive little or no feedback about how fast they eat in everyday settings.
Researchers already treat eating pace as a measurable signal. In laboratory and free-living studies, teams annotate “intake events” such as bites and sips on video, then compute metrics like bites per minute or burst patterns to study self-control, comfort, and energy intake. However, these analyses typically depend on manual labeling or bespoke tooling and are not accessible to consumers or digital-health partners. There is a gap between what research systems can measure about eating behavior and the kind of timely, privacy-preserving feedback that could help individuals make small but meaningful changes in daily life.

BitePulse AI investigates whether modern temporal deep learning can close part of this gap. The project explores if short meal videos, captured on a phone, can be converted automatically into bite and sip detections and an interpretable eating-pace summary that is returned to the user in under one minute. The primary end users are individual consumers who might benefit from gentle, real-time coaching about pace, and wellness organizations who may want objective but non-intrusive indicators of eating behavior for their programs. For consumers, the envisioned experience is a simple application that accepts a brief clip and returns a pace score, a timeline of detected intake events, and one sentence of neutral guidance. For partners, the same signals can be exposed through an API without exposing or storing identifiable video.

To support this investigation, we use the EatSense dataset, a public collection of real-world meal videos with anonymized faces and frame-level labels for eating, chewing, drinking, and resting (University of Edinburgh, School of Informatics, n.d.). These annotations enable supervised learning of window-level “intake versus non-intake” predictions and aggregation into event-level bite and sip detections. In a deployed system, analogous data would come from user-recorded clips, processed either entirely on device or in a secure backend that discards raw video after inference.

The technical approach is to construct time-aligned windows from the annotated segments and train temporal models that map windows to binary intake labels. A Temporal Convolutional Network (TCN) operating on pose features serves as a lightweight baseline that is compatible with on-device inference. A compact 3D convolutional neural network (3D-CNN) over RGB clips adds appearance cues such as utensil contact and food visibility. We evaluate both window-level classification metrics and event-level precision–recall under intersection-over-union matching to ensure that model outputs correspond to meaningful intake events rather than isolated noisy frames.
The central hypothesis is that temporal models trained on labeled meal videos can achieve practically useful precision and recall for intake detection and produce stable, interpretable measures of eating pace that are suitable for real-time feedback. If this hypothesis is supported, BitePulse AI points toward a privacy-respecting, deployable “eating-pace coach” that brings methods currently used only in research labs into everyday life for consumers and wellness programs.

## **BitePulse AI Audience**
BitePulse AI serves both consumers (B2C) and wellness partners (B2B):
* For consumers (B2C), it helps everyday eaters who want better comfort after meals, calmer digestion, and more mindful control. A user can record a short meal clip and get clear feedback on their eating pace.
* For wellness partners (B2B), like nutrition coaches and digital health platforms, BitePulse AI delivers objective eating behavior signals (for example, slow pauses versus nonstop bursts) through an API, without anyone having to watch the video. In both cases, the goal is awareness and coaching, not medical diagnosis.

## **Data Summary**

The BitePulse AI prototype is built on the [EatSense dataset](https://groups.inf.ed.ac.uk/vision/DATASETS/EATSENSE/), a public collection of real-world meal videos with anonymized faces and frame-level activity labels. Each recording contains RGB video of a person eating at a table and a set of time-aligned annotations describing what the person is doing in each moment. For this project, the most important labels are eating, drinking, chewing, and resting, which we use to define intake versus non-intake behavior.
At a high level there are four layers of variables. At the session level, each meal has an identifier, basic context (for example, lunch versus snack), and camera setup. At the frame level, each image has a timestamp, a frame index, and pre-computed 2D body-pose key points for head, torso, arms, and hands. At the segment level, the dataset provides start and end times for labeled activities, such as a chewing phase or a drink. On top of this, our capstone introduces a window level: we slide a fixed-length window (for example, 0.5 seconds) over time with a fixed stride and assign each window a binary label indicating whether it contains an intake event.

The window representation is a key part of the novelty in our approach. Rather than train directly on variable-length segments, we create a uniform “grid” over time that can be used consistently across pose and RGB models. Each window carries (a) a short sequence of pose features, engineered from the 2D key points as relative positions to the head and simple velocities, and (b) an index into the original video frames so that the same window can be mapped to an RGB clip for the 3D-CNN. The target label for each window is derived from the segment annotations using an overlap rule: a window is positive if its time span overlaps an eating or drinking segment beyond a chosen threshold, and negative otherwise. This produces a single table of training examples that sits between the raw dataset and our models and makes it straightforward to compare pose-only and RGB baselines.

Like most behavioral datasets, EatSense is not perfectly regular. There are occasional missing or low-confidence pose detections, small gaps in the activity labels, and slight misalignments between annotation timestamps and video frame times. Some participants move partially out of frame or occlude their face with hands or utensils. In addition, intake events are relatively rare compared to background motion, so the raw window distribution is highly imbalanced. To handle these issues, we derive labels using timestamps rather than frame indices, drop windows that have no valid pose information, and merge very short gaps in intake segments so that a single bite is not split into multiple tiny labels. On the modeling side we compensate for class imbalance with a combination of class weighting in the loss and balanced sampling of positive and negative windows during training.

Exploratory analysis of the window table reveals several patterns that connect directly to the project goal. Positive intake windows are short, often clustered around brief periods where the hand moves rapidly toward the mouth and then away, while the head remains relatively stable. Below chart illustrates this for a sample meal: the wrist trajectories show compact, directed paths toward the face during intake, contrasted with smaller, more diffuse motion when the participant is resting or chewing without new intake.
<p align="center">
  <img src="https://github.com/user-attachments/assets/5d011134-730f-414a-a410-0344785e8854"  
       alt="Strivio logo" width="400" height="400" />
</p>

When we look at the same sequence in time, we see that intake moments correspond to sharp peaks in wrist speed and characteristic changes in elbow angle. below chart shows wrist speed and elbow angle over the first five seconds of the same clip. Short bursts of high wrist velocity align with rapid elbow flexion as the utensil approaches the mouth, followed by a slower return. These patterns motivate the use of temporal convolution rather than purely frame-wise models and suggest that pose-only features should be sufficient for a strong baseline.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a232f79d-4398-4637-8df8-d20ef1bddd54"   
       alt="Strivio logo" width="700" height="400" />
</p>

We also observe correlations and redundancies within the raw pose coordinates. Neighboring key points and their absolute image positions are strongly correlated, especially within each limb. To keep the model focused on behavior rather than camera geometry, we express pose features in a head-centered coordinate frame and include simple temporal derivatives instead of a large set of raw coordinates. This reduces input dimensionality and helps the Temporal Convolutional Network focus on relative motion patterns that generalize across subjects and camera setups. For the RGB path we use the same window index to extract short clips of frames, which allows a 3D-CNN to learn complementary appearance
cues such as utensil type, cup orientation, or partial occlusions that are not visible in pose.

Taken together, the dataset and these engineered variables give us three views of the same underlying behavior: (1) raw video, (2) anonymized, structured pose sequences, and (3) a
regular grid of labeled windows that bridge between them. This design enables fair comparison between pose-based and RGB models, event-level evaluation built from window outputs, and a direct mapping from model predictions to the bite timeline and pace metrics that drive the BitePulse AI user experience.

## **Methods Used**

* Temporal windowing over frame-level EatSense labels  
* Pose-based Temporal Convolutional Network (TCN) for intake vs. non-intake
* RGB 3D-CNN on short frame clips for appearance cues
* Event-level aggregation from window scores (IoU-based matching, non-max suppression)
* Class imbalance handling via weighted BCE loss and balanced sampling
* Evaluation with window-level metrics (accuracy, precision, recall, F1) and event-level PR curves

## **Technologies**
* Python 3 (Google Colab)
* PyTorch & Torchvision
* NumPy & pandas
* Matplotlib for visualization
* OpenCV / FFmpeg for video-to-frames extraction
* Jupyter/Colab notebooks for experimentation and reports

## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## **Acknowledgments**

Thanks to Professor Anna Marbut for guidance and feedback.
