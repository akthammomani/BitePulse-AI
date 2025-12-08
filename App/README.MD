# BitePulse AI

Real-time bite detection & eating-pace insights — all on your device via the browser camera.  
BitePulse AI passively detects **intake gestures** (hand-to-mouth) and turns them into actionable feedback like **Bites per Minute (BPM)**, pauses, and simple coaching cues (e.g., *Slow*, *Typical*, *Fast*).

> **Privacy:** The app runs computer-vision analysis locally in the session; no video frames are stored or uploaded by the app. WebRTC is used only to get the live stream from your camera.

---

## Table of contents

- [Demo & quick start](#demo--quick-start)
- [What it measures](#what-it-measures)
- [How it works (high level)](#how-it-works-high-level)
- [Models & algorithms](#models--algorithms)
- [Why we didn’t ship MS-TCN (yet)](#why-we-didnt-ship-ms-tcn-yet)
- [Repository layout](#repository-layout)
- [Local development](#local-development)
- [Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
- [TURN/STUN configuration](#turnstun-configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License & acknowledgments](#license--acknowledgments)

---

## Demo & quick start

1. **Open the app** (local: `streamlit run app/bitepulse_app.py` or your Streamlit Cloud URL).
2. **Allow camera** permissions in your browser when prompted.
3. Click **Start** under **Live Analysis**.  
4. Eat normally. The app will:
   - show **INTAKE / NON_INTAKE** state,
   - draw mouth and wrist markers,
   - increment bites and compute BPM/pauses/percent intake,
   - update charts in the **Session Statistics** panel.

---

## What it measures

All metrics are computed per session and update continuously:

- **Duration** (`duration_str`)  
  Time since the session started.

- **Bites** (`bites`)  
  Count of detected bite events.

- **BPM (session average)** Bites per minute over the whole session:  

  $$\mathrm{BPM} = \frac{B}{T/60} = 60 \times \frac{B}{T}$$

  Where $B$ is the total number of detected bites and $T$ is session duration in seconds.

- **Rolling BPM (30s)** For each second $t$, count bites in the last 30 seconds and scale to per-minute:  

  $$\mathrm{rolling\_bpm}(t)= \frac{\left|\{\, b_i \mid t-30 < b_i \le t \,\}\right|}{30/60}= 2 \, \left|\{\, b_i \mid t-30 < b_i \le t \,\}\right|$$


  Here $b_i$ are bite timestamps in seconds.

- **% Intake (`intake_pct`)** Share of frames flagged as “intake”:  
  $$\mathrm{intake} = 100 \times \frac{\sum_{k=1}^{N}\mathrm{intake\_flag}_k}{N}$$
  Where $\mathrm{intake\_flag}_k \in \{0,1\}$ for frame $k$ and $N$ is the number of frames.

- **Pauses** (`pause_count`)  
  Number of long gaps between bites, where $\Delta t >$ `PAUSE_THRESHOLD_SEC` (default **10s**).

- **IBI (Inter-Bite Interval) histogram** Distribution of $\Delta t$ between consecutive bites; we show median and 75th percentile.
- **Pace label** (`pace_label`)  
  Based on session-average BPM:
  - `< 3` → **SLOWER**
  - `3 – 7` → **TYPICAL**
  - `> 7` → **FASTER**

- **First/Second-half BPM**  
  BPM in the first and second halves of the session duration; useful for drift.

- **Left/Right hand %**  
  Portion of bites attributed to the closer wrist at the time of the bite.

---

## How it works (high level)

1. **WebRTC camera stream** → the browser grants the app access to your webcam.
2. **Face & pose landmarks** → MediaPipe tracks **lips** and **wrists** per frame.
3. **Intake frame** → if the closest wrist is within a distance threshold to the mouth region.
4. **Bite event** → one bite is registered when an intake run lasts at least *N* consecutive frames.
5. **Continuous analytics** → compute BPM, % intake, IBI, pauses, hand split, and draw an overlay.

Everything is computed in real time; there’s no batch post-processing step.

---

## Models & algorithms

### 1) Landmark detection (pretrained, on-device)
- **MediaPipe Face Mesh** — extracts dense face landmarks; we use the mouth/lips indices to build a mouth box and center.  
- **MediaPipe Pose** — provides wrist landmarks (left/right) with a **visibility** score.

> **Why MediaPipe?**  
> Lightweight, fast, multi-platform, and battle-tested for webcam scale. It’s sufficient for a geometric proximity heuristic.

### 2) Intake frame rule (geometry)
- Compute Euclidean distance from the nearest wrist to the mouth center.  
- If `distance < mouth_hand_dist_thresh` (default ≈ **105** pixels at 640×360) → **intake frame**.

### 3) Bite detector (temporal smoothing)
- Maintain a counter of **consecutive intake frames**.  
- When the run length crosses `min_bite_frames` (default **4**) and then ends → **+1 bite**.  
  This filters brief occlusions and random hand motion.

### 4) Metrics
- See formulas in [What it measures](#what-it-measures).  
- We also track **hand used** (nearest wrist at bite moment) and **rolling BPM** over a 30-second window for short-term pace.

---

## Why we didn’t ship MS-TCN (yet)

We explored **MS-TCN** (Multi-Stage Temporal Convolutional Network) for sequence labeling because it’s strong on action segmentation. We decided **not** to ship it in v1 for the following reasons:

1. **Data dependency** — MS-TCN needs **frame-level bite annotations** across diverse cameras, angles, utensils, lighting, hand dominance, foods, etc. We don’t yet have a large, well-labeled dataset that generalizes across those variables.
2. **Latency & streaming** — Classic MS-TCN works best with access to a **temporal window** (or even whole sequences). For real-time, low-latency feedback, we prefer a **causal** and lightweight detector that reacts in a few frames.
3. **Complexity vs. v1 value** — The geometric proximity + short run-length heuristic reaches good practical accuracy on typical laptop webcams, with minimal compute and no separate model server.
4. **Maintenance & deployment** — Keeping inference entirely in the Streamlit runtime (no external model service) simplifies ops, especially on Streamlit Cloud where **TURN/STUN** and browser environments already add complexity.

**Plan:** keep the current approach as a robust, low-latency baseline and later add an **optional TCN-lite** (causal 1D temporal conv) trained on collected telemetry that can run online.

---

## Repository layout


