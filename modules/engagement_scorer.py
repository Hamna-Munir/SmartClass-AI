import cv2
import numpy as np
import pickle
import mediapipe as mp

# ── Load trained focus model
with open("models/focus_model.pkl", "rb") as f:
    focus_model = pickle.load(f)

# ── Landmark indices
LEFT_EYE   = [33,  160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ── Shared MediaPipe FaceMesh — lazy module-level singleton.
# FaceMesh itself is a stateless inference model: it doesn't remember
# which face it processed last, so it's safe (and much faster) for
# EVERY EngagementScorer instance to share this ONE model, instead of
# each instance building its own private copy. This matters most in
# batch flows (Class Analysis) where the app used to create one
# EngagementScorer — and therefore one full FaceMesh model load —
# PER STUDENT PHOTO. With 20-30 students that was 20-30 redundant
# model loads; now it's exactly one, no matter how many scorers exist.
#
# What must NOT be shared — and isn't — is per-student state:
# blink_count, _prev_ear, and _history all still live on `self` inside
# each EngagementScorer instance below, so scores for one student can
# never leak into another's.
_shared_mesh = None

def _get_shared_mesh():
    global _shared_mesh
    if _shared_mesh is None:
        _shared_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _shared_mesh

def _ear(lm, idx, w, h):
    """Eye Aspect Ratio"""
    p = [(lm[i].x*w, lm[i].y*h) for i in idx]
    A = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    B = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    C = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (A+B)/(2.0*C+1e-6)

def _head_pose(lm):
    """Check if student is looking forward"""
    nose_x     = lm[1].x
    l_ear_x    = lm[234].x
    r_ear_x    = lm[454].x
    face_mid_x = (l_ear_x + r_ear_x) / 2
    h_tilt     = abs(nose_x - face_mid_x)

    nose_y     = lm[1].y
    chin_y     = lm[152].y
    fore_y     = lm[10].y
    face_mid_y = (fore_y + chin_y) / 2
    v_tilt     = abs(nose_y - face_mid_y)

    looking_forward = h_tilt < 0.08 and v_tilt < 0.12
    return looking_forward, round(h_tilt, 3), round(v_tilt, 3)

class EngagementScorer:
    def __init__(self):
        # No per-instance model here anymore — _get_mesh() below
        # returns the shared singleton. Only per-student state lives
        # on self.
        self.blink_count = 0
        self._prev_ear   = 0.3
        self._history    = []

    def _get_mesh(self):
        return _get_shared_mesh()

    def score(self, frame, emotion="Neutral"):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            res = self._get_mesh().process(rgb)
        except Exception:
            return self._default()

        if not res.multi_face_landmarks:
            return self._default()

        lm = res.multi_face_landmarks[0].landmark

        # ── 6 features jo model expect karta hai
        le  = _ear(lm, LEFT_EYE,  w, h)
        re  = _ear(lm, RIGHT_EYE, w, h)
        avg = (le + re) / 2.0

        lw = np.linalg.norm(
            np.array([lm[LEFT_EYE[0]].x*w,  lm[LEFT_EYE[0]].y*h]) -
            np.array([lm[LEFT_EYE[3]].x*w,  lm[LEFT_EYE[3]].y*h])
        )
        rw = np.linalg.norm(
            np.array([lm[RIGHT_EYE[0]].x*w, lm[RIGHT_EYE[0]].y*h]) -
            np.array([lm[RIGHT_EYE[3]].x*w, lm[RIGHT_EYE[3]].y*h])
        )
        ed = np.linalg.norm(
            np.array([lm[33].x*w,  lm[33].y*h]) -
            np.array([lm[362].x*w, lm[362].y*h])
        )

        features = np.array([[le, re, avg, lw, rw, ed]])

        # ── Blink detect
        if self._prev_ear > 0.25 and avg < 0.18:
            self.blink_count += 1
        self._prev_ear = avg

        # ── Head pose
        looking_forward, h_tilt, v_tilt = _head_pose(lm)

        # ── Model prediction — actual trained model
        prediction  = int(focus_model.predict(features)[0])
        proba       = focus_model.predict_proba(features)[0]
        focus_prob  = float(proba[1])  # probability of class 1 (Focused)

        # ── Engagement score 0-100
        base_score = int(focus_prob * 100)

        # Head pose adjustment
        if not looking_forward:
            penalty    = min(25, int(h_tilt * 200 + v_tilt * 100))
            base_score = max(0, base_score - penalty)

        # Blink adjustment
        blink_penalty = min(10, self.blink_count)
        base_score    = max(0, base_score - blink_penalty)

        # Rolling average — smooth results
        self._history.append(base_score)
        if len(self._history) > 8:
            self._history.pop(0)
        smoothed = int(sum(self._history) / len(self._history))

        # ── State classification
        if prediction == 1 and smoothed >= 60:
            state = "Highly Focused"
        elif prediction == 1:
            state = "Focused"
        elif prediction == 0 and smoothed >= 35:
            state = "Moderately Focused"
        else:
            state = "Distracted"

        return {
            "engagement_score": smoothed,
            "raw_focus_prob":   round(focus_prob * 100, 1),
            "prediction":       prediction,   # 0=Distracted, 1=Focused
            "state":            state,
            "ear":              round(avg, 3),
            "looking_forward":  looking_forward,
            "h_tilt":           h_tilt,
            "v_tilt":           v_tilt,
            "blinks":           self.blink_count,
            "alert":            smoothed < 35,
            "raw_scores": {
                "ear_score":   int(avg / 0.32 * 100),
                "pose_score":  100 if looking_forward else max(0, 100 - int(h_tilt*200)),
                "model_score": int(focus_prob * 100)
            }
        }

    def _default(self):
        return {
            "engagement_score": 0,
            "raw_focus_prob":   0.0,
            "prediction":       0,
            "state":            "No Face",
            "ear":              0.0,
            "looking_forward":  False,
            "h_tilt":           0.0,
            "v_tilt":           0.0,
            "blinks":           self.blink_count,
            "alert":            False,
            "raw_scores":       {"ear_score":0,"pose_score":0,"model_score":0}
        }