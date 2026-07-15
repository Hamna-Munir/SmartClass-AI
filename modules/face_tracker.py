# face_tracker.py — YOLO based multi-face detection
# Developer: Hamna Munir | Supervisor: Sir Nadeem

import cv2
import numpy as np
import pickle

with open("models/focus_model.pkl", "rb") as f:
    focus_model = pickle.load(f)

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EMOTION_WEIGHTS = {
    "Happy":1.15,"Neutral":1.0,"Surprise":0.9,
    "Sad":0.65,"Fear":0.55,"Angry":0.45,
    "Disgust":0.35,"N/A":0.8
}

# ── Lazy globals
_face_mesh  = None
_yolo_model = None

def _get_mesh():
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    return _face_mesh

def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n-face.pt")
        except Exception:
            _yolo_model = "failed"
    return _yolo_model if _yolo_model != "failed" else None

def _ear(lm, idx, w, h):
    p = [(lm[i].x*w, lm[i].y*h) for i in idx]
    A = np.linalg.norm(np.array(p[1])-np.array(p[5]))
    B = np.linalg.norm(np.array(p[2])-np.array(p[4]))
    C = np.linalg.norm(np.array(p[0])-np.array(p[3]))
    return (A+B)/(2.0*C+1e-6)

def _head_pose(lm):
    nose_x     = lm[1].x
    l_ear_x    = lm[234].x
    r_ear_x    = lm[454].x
    face_mid_x = (l_ear_x+r_ear_x)/2
    h_tilt     = abs(nose_x-face_mid_x)
    nose_y     = lm[1].y
    chin_y     = lm[152].y
    fore_y     = lm[10].y
    face_mid_y = (fore_y+chin_y)/2
    v_tilt     = abs(nose_y-face_mid_y)
    looking    = h_tilt<0.1 and v_tilt<0.15
    return looking, round(h_tilt,3), round(v_tilt,3)

def _score_face(le, re, avg, lw, rw, ed, looking, emotion="Neutral"):
    try:
        features   = np.array([[le,re,avg,lw,rw,ed]])
        prediction = int(focus_model.predict(features)[0])
        proba      = focus_model.predict_proba(features)[0]
        focus_prob = float(proba[1])
        base       = int(focus_prob*100)
        if not looking: base = max(0, base-20)
        ew   = EMOTION_WEIGHTS.get(emotion,0.8)
        base = max(0, min(100, int(base*ew)))
        if prediction==1 and base>=60:   state="Highly Focused"
        elif prediction==1:              state="Focused"
        elif prediction==0 and base>=35: state="Moderate"
        else:                            state="Distracted"
        return base, prediction, state, round(focus_prob*100,1)
    except Exception:
        return 50, 1, "Moderate", 50.0

def _analyze_face_crop(crop, emotion="Neutral"):
    """Analyze a single face crop using MediaPipe"""
    if crop is None or crop.size==0:
        return None

    h, w = crop.shape[:2]
    if h < 20 or w < 20:
        return None

    # Resize crop — MediaPipe needs decent size
    scale  = max(1, 120//min(h,w))
    crop_r = cv2.resize(crop, (w*scale, h*scale))
    hr, wr = crop_r.shape[:2]

    rgb = cv2.cvtColor(crop_r, cv2.COLOR_BGR2RGB)
    try:
        res = _get_mesh().process(rgb)
    except Exception:
        return None

    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark

    le  = _ear(lm, LEFT_EYE,  wr, hr)
    re  = _ear(lm, RIGHT_EYE, wr, hr)
    avg = (le+re)/2.0

    lw_px = np.linalg.norm(
        np.array([lm[LEFT_EYE[0]].x*wr, lm[LEFT_EYE[0]].y*hr]) -
        np.array([lm[LEFT_EYE[3]].x*wr, lm[LEFT_EYE[3]].y*hr]))
    rw_px = np.linalg.norm(
        np.array([lm[RIGHT_EYE[0]].x*wr, lm[RIGHT_EYE[0]].y*hr]) -
        np.array([lm[RIGHT_EYE[3]].x*wr, lm[RIGHT_EYE[3]].y*hr]))
    ed_px = np.linalg.norm(
        np.array([lm[33].x*wr, lm[33].y*hr]) -
        np.array([lm[362].x*wr, lm[362].y*hr]))

    looking, h_tilt, v_tilt = _head_pose(lm)
    score, pred, state, prob = _score_face(
        le, re, avg, lw_px, rw_px, ed_px, looking, emotion)

    return {
        "score":    score,
        "pred":     pred,
        "state":    state,
        "prob":     prob,
        "ear":      round(avg,3),
        "looking":  looking,
        "h_tilt":   h_tilt,
        "v_tilt":   v_tilt,
    }

def _detect_faces_opencv(frame):
    """OpenCV multi-scale face detection — fallback"""
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(25,25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(boxes)==0:
        return []
    return [(x,y,x+w,y+h) for x,y,w,h in boxes]

def _detect_faces_yolo(frame):
    """YOLO face detection — best for classroom"""
    model = _get_yolo()
    if model is None:
        return []
    try:
        results = model(frame, conf=0.25, iou=0.4, verbose=False)
        boxes   = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                boxes.append((x1,y1,x2,y2))
        return boxes
    except Exception:
        return []

def _detect_faces_dnn(frame):
    """OpenCV DNN — better than cascade"""
    try:
        h,w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300,300),
            (104.0,177.0,123.0)
        )
        net_path = cv2.data.haarcascades.replace(
            "haarcascades/","dnn/")
        prototxt = net_path+"deploy.prototxt"
        model_f  = net_path+"res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNetFromCaffe(prototxt, model_f)
        net.setInput(blob)
        dets  = net.forward()
        boxes = []
        for i in range(dets.shape[2]):
            conf = dets[0,0,i,2]
            if conf>0.4:
                box = dets[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype(int)
                boxes.append((
                    max(0,x1), max(0,y1),
                    min(w,x2), min(h,y2)
                ))
        return boxes
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════
def analyze_classroom(frame, emotions_per_face=None):
    """
    Detect ALL faces in classroom photo and analyze each one.
    Uses YOLO → OpenCV cascade → MediaPipe fallback chain.
    """
    H, W   = frame.shape[:2]
    students  = []
    annotated = frame.copy()

    # ── Step 1: Detect all face boxes
    # Try YOLO first (best for classroom)
    boxes = _detect_faces_yolo(frame)
    method = "YOLO"

    # Fallback — OpenCV cascade with preprocessing
    if len(boxes) == 0:
        # Enhance image first
        enhanced = _enhance_image(frame)
        boxes    = _detect_faces_opencv(enhanced)
        method   = "OpenCV"

    # Fallback — MediaPipe whole image
    if len(boxes) == 0:
        boxes  = _detect_faces_mediapipe_full(frame)
        method = "MediaPipe"

    if len(boxes) == 0:
        cv2.putText(annotated,
                   "No faces detected — try clearer photo",
                   (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0,0,255), 2)
        return students, annotated

    # ── Step 2: Analyze each face
    for idx, (x1,y1,x2,y2) in enumerate(boxes):
        # Add padding around face
        pad  = int(max(x2-x1, y2-y1)*0.15)
        fx1  = max(0, x1-pad)
        fy1  = max(0, y1-pad)
        fx2  = min(W, x2+pad)
        fy2  = min(H, y2+pad)
        crop = frame[fy1:fy2, fx1:fx2]

        emotion = (emotions_per_face or {}).get(idx,"Neutral")
        result  = _analyze_face_crop(crop, emotion)

        if result is None:
            # Assign default moderate score if analysis fails
            result = {
                "score":50,"pred":1,"state":"Moderate",
                "prob":50.0,"ear":0.25,"looking":True,
                "h_tilt":0.05,"v_tilt":0.05
            }

        sid = f"S{idx+1}"
        students.append({
            "id":             sid,
            "index":          idx,
            "engagement":     result["score"],
            "focus_prob":     result["prob"],
            "prediction":     result["pred"],
            "state":          result["state"],
            "emotion":        emotion,
            "ear":            result["ear"],
            "looking_forward":result["looking"],
            "h_tilt":         result["h_tilt"],
            "v_tilt":         result["v_tilt"],
            "alert":          result["score"]<35,
            "face_box":       (fx1,fy1,fx2-fx1,fy2-fy1),
        })

        # ── Draw on frame
        score = result["score"]
        if score>=60:   c=(21,128,61)
        elif score>=35: c=(180,83,9)
        else:           c=(220,38,38)

        cv2.rectangle(annotated,(fx1,fy1),(fx2,fy2),c,2)

        # Label background
        label   = f"{sid}:{score}%"
        (lw2,lh2),_ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(fy1-6, lh2+4)
        cv2.rectangle(annotated,
                     (fx1,ly-lh2-4),(fx1+lw2+6,ly+2),c,-1)
        cv2.putText(annotated, label,
                   (fx1+3,ly),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,(255,255,255),1)

        # State below box
        cv2.putText(annotated, result["state"],
                   (fx1, fy2+16),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, c, 1)

    # ── Summary overlay
    total   = len(students)
    avg_eng = int(sum(s["engagement"] for s in students)/total) if total else 0
    focused = sum(1 for s in students if s["prediction"]==1)
    dist    = total-focused

    summary_txt = f"Detected:{total} | Avg:{avg_eng}% | Focused:{focused} | Dist:{dist} [{method}]"
    sw = cv2.getTextSize(summary_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    cv2.rectangle(annotated,(8,8),(sw+16,28),(15,23,42),-1)
    cv2.putText(annotated, summary_txt,
               (12,22), cv2.FONT_HERSHEY_SIMPLEX,
               0.45,(255,255,255),1)

    return students, annotated


def _enhance_image(frame):
    """Enhance image quality for better face detection"""
    # Upscale if too small
    h,w = frame.shape[:2]
    if max(h,w) < 800:
        scale  = 800/max(h,w)
        frame  = cv2.resize(frame,
                           (int(w*scale),int(h*scale)),
                           interpolation=cv2.INTER_CUBIC)

    # CLAHE — contrast enhancement
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    l     = clahe.apply(l)
    lab   = cv2.merge([l,a,b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen
    kernel   = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    enhanced = cv2.filter2D(enhanced,-1,kernel)
    return enhanced


def _detect_faces_mediapipe_full(frame):
    """Use MediaPipe on full image"""
    H,W   = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        import mediapipe as mp
        mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=30,
            refine_landmarks=False,
            min_detection_confidence=0.3,
        )
        res   = mesh.process(rgb)
        mesh.close()
        if not res.multi_face_landmarks:
            return []
        boxes = []
        for fl in res.multi_face_landmarks:
            xs  = [l.x*W for l in fl.landmark]
            ys  = [l.y*H for l in fl.landmark]
            x1  = max(0,  int(min(xs))-10)
            y1  = max(0,  int(min(ys))-10)
            x2  = min(W,  int(max(xs))+10)
            y2  = min(H,  int(max(ys))+10)
            boxes.append((x1,y1,x2,y2))
        return boxes
    except Exception:
        return []


def get_class_summary(students):
    if not students:
        return {"total":0,"avg_score":0,"focused":0,"moderate":0,
                "distracted":0,"alert_count":0,"top_student":None,
                "low_student":None,"focus_pct":0}
    total      = len(students)
    avg_score  = int(sum(s["engagement"] for s in students)/total)
    focused    = sum(1 for s in students if s["engagement"]>=60)
    moderate   = sum(1 for s in students if 35<=s["engagement"]<60)
    distracted = sum(1 for s in students if s["engagement"]<35)
    alerts     = sum(1 for s in students if s["alert"])
    sorted_s   = sorted(students, key=lambda x: x["engagement"])
    return {
        "total":       total,
        "avg_score":   avg_score,
        "focused":     focused,
        "moderate":    moderate,
        "distracted":  distracted,
        "alert_count": alerts,
        "top_student": sorted_s[-1] if sorted_s else None,
        "low_student": sorted_s[0]  if sorted_s else None,
        "focus_pct":   round(focused/total*100) if total else 0,
    }