# app/bitepulse_app.py
import streamlit as st
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import cv2
import av
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ---------------------------
# Video/analysis parameters
# ---------------------------
OUT_W, OUT_H = 640, 360            # lower res = less bandwidth, more reliable on Cloud
PAUSE_THRESHOLD_SEC = 10.0

# ---------------------------
# ICE config (STUN + TURN)
# ---------------------------
def build_rtc_config(force_turn: bool) -> dict:
    """
    STUN always; TURN optional via st.secrets['turn'].
    If force_turn and TURN creds present -> iceTransportPolicy='relay'.
    """
    ice_servers = [
        {"urls": [
            "stun:stun.l.google.com:19302",
            "stun:stun1.l.google.com:19302",
            "stun:global.stun.twilio.com:3478",
            "stun:stun.cloudflare.com:3478",
        ]},
    ]
    turn = st.secrets.get("turn", {})
    turn_urls = [u for u in [turn.get("url_1"), turn.get("url_2")] if u]
    has_turn = bool(turn_urls and turn.get("username") and turn.get("credential"))

    if has_turn:
        ice_servers.append({
            "urls": turn_urls,
            "username": turn["username"],
            "credential": turn["credential"],
        })
        if force_turn:
            # Why: corporate firewalls often block UDP; forcing relay uses TURN over TCP/TLS 443
            return {"iceServers": ice_servers, "iceTransportPolicy": "relay"}

    return {"iceServers": ice_servers}

def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def draw_text_with_outline(img, text, x, y, font_scale, color, thickness=2):
    # Why: double stroke keeps text readable after WebRTC compression
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ---------------------------
# Mediapipe setup
# ---------------------------
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
POSE_LANDMARKS = mp_pose.PoseLandmark

MOUTH_LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
    81, 42, 183, 78
]

def get_mouth_box_and_center(face_landmarks, image_w, image_h):
    xs, ys = [], []
    for idx in MOUTH_LIPS_INDICES:
        lm = face_landmarks.landmark[idx]
        xs.append(lm.x * image_w)
        ys.append(lm.y * image_h)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return (int(x_min), int(y_min), int(x_max), int(y_max)), (cx, cy)

def get_hand_points_labeled(pose_landmarks, image_w, image_h) -> Dict[str, Tuple[float, float]]:
    hands = {}
    left_lm = pose_landmarks.landmark[POSE_LANDMARKS.LEFT_WRIST]
    if left_lm.visibility > 0.5:
        hands["Left"] = (left_lm.x * image_w, left_lm.y * image_h)
    right_lm = pose_landmarks.landmark[POSE_LANDMARKS.RIGHT_WRIST]
    if right_lm.visibility > 0.5:
        hands["Right"] = (right_lm.x * image_w, right_lm.y * image_h)
    return hands

def get_closest_hand(mouth_center, hand_dict):
    if not hand_dict or mouth_center is None:
        return None, None
    cx, cy = mouth_center
    min_dist = float('inf')
    closest_label = None
    for label, (hx, hy) in hand_dict.items():
        dist = np.hypot(hx - cx, hy - cy)
        if dist < min_dist:
            min_dist = dist
            closest_label = label
    if closest_label is None:
        return None, None
    return min_dist, closest_label

# ---------------------------
# Bite session (unchanged)
# ---------------------------
@dataclass
class BiteSession:
    start_time: Optional[float] = None
    frame_times: List[float] = field(default_factory=list)
    intake_flags: List[int] = field(default_factory=list)
    bite_timestamps: List[float] = field(default_factory=list)
    bite_hands: List[str] = field(default_factory=list)
    bite_count: int = 0
    intake_run: int = 0

    def update(self, is_intake: bool, is_bite_this_frame: bool, hand_used: Optional[str] = None):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time
        self.frame_times.append(t)
        self.intake_flags.append(1 if is_intake else 0)
        if is_bite_this_frame:
            self.bite_timestamps.append(t)
            self.bite_hands.append(hand_used if hand_used else "Unknown")
            self.bite_count += 1

    @property
    def duration_sec(self) -> float:
        return self.frame_times[-1] if self.frame_times else 0.0

    @property
    def intake_ratio(self) -> float:
        return (sum(self.intake_flags) / len(self.intake_flags)) if self.intake_flags else 0.0

# ---------------------------
# Video processor
# ---------------------------
class BitePulseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=False)
        self.face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.session = BiteSession()

        self.lock = threading.Lock()
        self.summary: Optional[Dict[str, Any]] = None

        self.mouth_hand_dist_thresh = 105
        self.min_bite_frames = 4

        self._consecutive_intake = 0
        self.last_min_dist = None
        self.last_status = "NON_INTAKE"
        self.last_detected_hand = None

    def _generate_summary(self) -> Dict[str, Any]:
        s = self.session
        duration_sec = s.duration_sec
        duration_str = f"{int(duration_sec)}s" if duration_sec < 60 else f"{int(duration_sec//60)}m {int(duration_sec%60)}s"
        duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
        bites = len(s.bite_timestamps)
        avg_bpm = (bites / duration_min) if duration_min > 0 else 0.0
        intake_pct = s.intake_ratio * 100.0

        bt = np.array(s.bite_timestamps)
        pause_count = int(np.sum(np.diff(bt) > PAUSE_THRESHOLD_SEC)) if bt.size >= 2 else 0

        total = len(s.bite_hands)
        if total:
            left_pct = 100.0 * s.bite_hands.count("Left") / total
            right_pct = 100.0 * s.bite_hands.count("Right") / total
        else:
            left_pct = right_pct = 0.0

        half = duration_sec / 2.0
        bites_1st = sum(1 for t in s.bite_timestamps if t <= half)
        d1 = (half / 60.0) if half > 0 else 0
        d2 = ((duration_sec - half) / 60.0) if duration_sec > half else 0
        bpm_1st = (bites_1st / d1) if d1 > 0 else 0.0
        bpm_2nd = ((len(s.bite_timestamps) - bites_1st) / d2) if d2 > 0 else 0.0

        pace = "SLOWER" if avg_bpm < 3 else ("TYPICAL" if avg_bpm <= 7 else "FASTER")
        return {
            "duration_str": duration_str,
            "duration_min": duration_min,
            "bites": bites,
            "avg_bpm": avg_bpm,
            "intake_pct": intake_pct,
            "pace_label": pace,
            "pause_count": pause_count,
            "first_half_bpm": bpm_1st,
            "second_half_bpm": bpm_2nd,
            "left_hand_pct": left_pct,
            "right_hand_pct": right_pct,
            "bite_timestamps": list(s.bite_timestamps),
            "bite_hands": list(s.bite_hands),
            "recommendation": f"Your overall pace is **{pace}**.",
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w, _ = img.shape
            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pose_res = self.pose.process(rgb)
            face_res = self.face_mesh.process(rgb)

            mouth_box = None
            mouth_center = None
            hand_dict = {}

            if face_res.multi_face_landmarks:
                face_lms = face_res.multi_face_landmarks[0]
                mouth_box, mouth_center = get_mouth_box_and_center(face_lms, w, h)

            if pose_res.pose_landmarks:
                hand_dict = get_hand_points_labeled(pose_res.pose_landmarks, w, h)

            is_intake_frame = False
            min_dist = None
            closest_hand_label = None

            if mouth_center and hand_dict:
                min_dist, closest_hand_label = get_closest_hand(mouth_center, hand_dict)
                if (min_dist is not None) and (min_dist < self.mouth_hand_dist_thresh):
                    is_intake_frame = True
                    self.last_detected_hand = closest_hand_label

            is_bite_this_frame = False
            if is_intake_frame:
                self._consecutive_intake += 1
            else:
                if self._consecutive_intake >= self.min_bite_frames:
                    is_bite_this_frame = True
                self._consecutive_intake = 0

            self.session.update(is_intake_frame, is_bite_this_frame, self.last_detected_hand)
            self.last_min_dist = min_dist
            self.last_status = "INTAKE" if is_intake_frame else "NON_INTAKE"

            with self.lock:
                self.summary = self._generate_summary()

            vis = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
            sx, sy = OUT_W / w, OUT_H / h

            if mouth_box:
                x1, y1, x2, y2 = mouth_box
                x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                color = (0, 255, 0) if is_intake_frame else (200, 200, 200)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            for label, (x, y) in hand_dict.items():
                px, py = int(x * sx), int(y * sy)
                color = (0, 255, 255) if label == self.last_detected_hand and is_intake_frame else (255, 255, 0)
                cv2.circle(vis, (px, py), 6, color, -1)

            summary = self.summary or {}
            duration_str = summary.get("duration_str", "0s")
            bites = len(self.session.bite_timestamps)
            duration_sec = self.session.duration_sec
            duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
            bpm = (bites / duration_min) if duration_min > 0 else 0.0
            status_color = (0, 255, 0) if is_intake_frame else (255, 255, 255)

            draw_text_with_outline(vis, f"Time: {duration_str}", 20, 40, 0.8, (0, 255, 255), 2)
            draw_text_with_outline(vis, f"{self.last_status}", 20, 80, 0.8, status_color, 2)
            draw_text_with_outline(vis, f"Bites: {bites}", 20, 120, 0.9, (0, 255, 0), 2)
            draw_text_with_outline(vis, f"BPM: {bpm:.1f}", 20, 160, 0.9, (0, 255, 0), 2)
            if min_dist is not None:
                draw_text_with_outline(vis, f"Dist: {int(min_dist)} px", 20, 200, 0.6, (200, 200, 200), 2)

            return av.VideoFrame.from_ndarray(vis, format="bgr24")
        except Exception as e:
            print(f"Error in recv: {e}")
            return frame

# ---------------------------
# Plotly helpers
# ---------------------------
def _rolling_bpm(bite_ts: List[float], duration_sec: float, window: float = 30.0) -> pd.DataFrame:
    if duration_sec <= 0:
        return pd.DataFrame(columns=["t", "bpm"])
    t_grid = np.arange(1, int(duration_sec) + 1, dtype=int)
    bt = np.array(bite_ts, dtype=float)
    counts = []
    for t in t_grid:
        left = t - window
        counts.append(np.sum((bt > left) & (bt <= t)))
    bpm = (np.array(counts) / (window / 60.0)) if window > 0 else np.zeros_like(counts)
    return pd.DataFrame({"t": t_grid, "bpm": bpm})

def _intake_per_second(frame_times: List[float], intake_flags: List[int]) -> pd.DataFrame:
    if not frame_times or not intake_flags:
        return pd.DataFrame(columns=["sec", "intake_ratio"])
    df = pd.DataFrame({"t": frame_times, "f": intake_flags})
    df["sec"] = df["t"].astype(int)
    agg = df.groupby("sec")["f"].mean().reset_index()
    agg.rename(columns={"f": "intake_ratio"}, inplace=True)
    return agg

def render_bite_tabs(summary: Dict[str, Any], session: BiteSession):
    bite_ts = summary["bite_timestamps"]
    if not bite_ts:
        st.write("No bites detected yet to plot.")
        return

    bites_idx = list(range(1, len(bite_ts) + 1))
    df_bites = pd.DataFrame({"t": bite_ts, "n": bites_idx})

    ibi = np.diff(bite_ts) if len(bite_ts) >= 2 else np.array([])
    pause_idx = np.where(ibi > PAUSE_THRESHOLD_SEC)[0]
    pause_lines = [bite_ts[i + 1] for i in pause_idx]

    duration_sec = summary["duration_min"] * 60.0
    df_roll = _rolling_bpm(bite_ts, duration_sec, window=30.0)
    df_intake = _intake_per_second(session.frame_times, session.intake_flags)

    bite_hands = summary.get("bite_hands", ["Unknown"] * len(bite_ts))

    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Rolling BPM (30s)", "Intervals (IBI)", "Hands"])

    with tab1:
        df_cum = df_bites.copy()
        fig = go.Figure()
        if not df_intake.empty:
            fig.add_trace(go.Scatter(
                x=df_intake["sec"], y=df_intake["intake_ratio"],
                mode="lines", fill="tozeroy", opacity=0.2, name="Intake ratio (per s)",
                hovertemplate="t=%{x}s<br>intake=%{y:.2f}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=df_cum["t"], y=df_cum["n"], mode="lines", name="Cumulative bites",
            hovertemplate="t=%{x:.1f}s<br>bites=%{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_bites["t"], y=df_bites["n"], mode="markers", name="Bites",
            marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
            hovertemplate="t=%{x:.1f}s<br>#%{y}<extra></extra>",
        ))
        for x in pause_lines:
            fig.add_vline(x=x, line_width=1, line_dash="dot", opacity=0.6)
        x_max = max(bite_ts[-1], 5)
        y_max = max(df_bites["n"].max(), 2)
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.08)", range=[0, x_max + 1]),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,.08)", range=[0, y_max + 0.5]),
            xaxis_title="Session Time (sec)",
            yaxis_title="Cumulative Bites",
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if df_roll.empty:
            st.write("Not enough data yet for rolling BPM.")
        else:
            fig = px.line(df_roll, x="t", y="bpm",
                          labels={"t": "Session Time (sec)", "bpm": "BPM (last 30s)"},
                          height=280)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if ibi.size == 0:
            st.write("Need at least 2 bites for intervals.")
        else:
            df_ibi = pd.DataFrame({"ibi_sec": ibi})
            median = float(np.median(ibi))
            p75 = float(np.percentile(ibi, 75))
            fig = px.histogram(df_ibi, x="ibi_sec",
                               nbins=min(20, max(5, len(ibi)//2)),
                               labels={"ibi_sec": "Seconds between bites"},
                               height=280)
            fig.add_vline(x=PAUSE_THRESHOLD_SEC, line_width=1, line_dash="dot", opacity=0.6)
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Median IBI: **{median:.1f}s**, 75th pct: **{p75:.1f}s**. Dotted line = pause threshold ({PAUSE_THRESHOLD_SEC:.0f}s).")

    with tab4:
        df_h = pd.DataFrame({"t": bite_ts, "n": bites_idx, "hand": bite_hands})
        fig = px.scatter(df_h, x="t", y="n", color="hand",
                         labels={"t": "Session Time (sec)", "n": "Bite #"},
                         height=280)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="BitePulse AI", layout="wide")

st.markdown(
    """
    <style>
      .kpi-card {border:1px solid rgba(0,0,0,.08);border-radius:12px;padding:10px 12px;background:rgba(255,255,255,.7);box-shadow:0 1px 2px rgba(0,0,0,.04);}
      .kpi-label {font-size:11px;font-weight:700;color:#6b7280;margin:0 0 6px 0;letter-spacing:.2px;}
      .kpi-value {font-size:16px;font-weight:800;color:#111827;margin:0;}
      .kpi-sub {font-size:12px;color:#6b7280;font-weight:600;margin-top:2px;}
      .section-subtitle {font-weight:700;font-size:14px;margin:6px 0 8px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Control: Force TURN (relay)
with st.sidebar:
    st.subheader("Connection")
    force_turn = st.toggle("Force TURN (relay only)", value=True, help="Requires TURN credentials in Secrets. Fixes strict NAT/firewalls.")
    st.caption("Add TURN creds in Cloud → Settings → Secrets:\n\n"
               "[turn]\nurl_1 = \"turn:global.relay.metered.ca:80\"\nurl_2 = \"turns:global.relay.metered.ca:443?transport=tcp\"\nusername = \"YOUR_USER\"\ncredential = \"YOUR_PASS\"")

# 40/60 layout
left_col, right_col = st.columns([4, 6])

with left_col:
    st.subheader("Live Analysis")

    webrtc_ctx = webrtc_streamer(
        key="bitepulse-live-intake",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=BitePulseProcessor,
        media_stream_constraints={
            "video": {
                "width":  {"min": OUT_W, "ideal": OUT_W, "max": OUT_W},
                "height": {"min": OUT_H, "ideal": OUT_H, "max": OUT_H},
                "frameRate": {"min": 15, "ideal": 24, "max": 24},
                "facingMode": "user",
            },
            "audio": False,
        },
        rtc_configuration=build_rtc_config(force_turn),  # <<< critical
        async_processing=True,
        video_html_attrs={
            "style": {
                "width": "100%",
                "height": "100%",
                "objectFit": "cover",
                "border": "0",
                "borderRadius": "10px"
            },
            "controls": False,
            "autoPlay": True,
        },
    )

    # ICE status banner
    state = getattr(webrtc_ctx, "state", None)
    conn_state = getattr(state, "connection_state", None)
    ice_state = getattr(state, "ice_connection_state", None)
    if ice_state in ("failed", "disconnected", "closed"):
        st.error(f"ICE state: {ice_state}. Enable 'Force TURN' and add TURN secrets.")
    elif ice_state in ("checking", "new"):
        st.warning(f"ICE state: {ice_state} (negotiating...)")
    elif ice_state == "connected":
        st.success("ICE state: connected")

with right_col:
    st.subheader("Session Statistics")
    right_placeholder = st.empty()

    def _kpi(label: str, value: str, sub: Optional[str] = None):
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              {sub_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_dashboard(summary: Optional[Dict[str, Any]], session: Optional[BiteSession]):
        with right_placeholder.container():
            col_a, col_b = st.columns(2)
            if not summary or "bites" not in summary:
                with col_a:
                    _kpi("Duration", "—")
                    _kpi("Bites", "—")
                    _kpi("BPM", "—")
                    _kpi("% Intake", "—")
                with col_b:
                    _kpi("Pace", "—")
                    _kpi("Pauses", "—")
                    _kpi("1st/2nd BPM", "—")
                    _kpi("L/R Hand", "—")
                st.info("Waiting for data... Start the camera and eat to see stats.")
                return

            with col_a:
                _kpi("Duration", summary['duration_str'])
                _kpi("Bites", f"{summary['bites']}")
                _kpi("BPM", f"{summary['avg_bpm']:.1f}")
                _kpi("% Intake", f"{int(summary['intake_pct'])}%")
            with col_b:
                _kpi("Pace", summary['pace_label'])
                _kpi("Pauses", f"{summary['pause_count']}")
                _kpi("1st/2nd BPM", f"{summary['first_half_bpm']:.1f}/{summary['second_half_bpm']:.1f}")
                _kpi("L/R Hand", f"{int(summary['left_hand_pct'])}%/{int(summary['right_hand_pct'])}%")

            st.markdown('<div class="section-subtitle">Charts</div>', unsafe_allow_html=True)
            render_bite_tabs(summary, session if session else BiteSession())

# Data pump
vp = getattr(webrtc_ctx, "video_processor", None)
summary = None
session_ref = None
if vp is not None:
    with vp.lock:
        summary = dict(vp.summary) if vp.summary is not None else None
        session_ref = vp.session

if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = None

summary_to_show = summary or st.session_state.get("last_summary")
with right_col:
    render_dashboard(summary_to_show, session_ref)
if summary:
    st.session_state["last_summary"] = summary

if getattr(getattr(webrtc_ctx, "state", None), "playing", False):
    time.sleep(0.2)
    _safe_rerun()
