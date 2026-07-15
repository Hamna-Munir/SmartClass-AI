import numpy as np
import cv2
import onnxruntime as ort

try:
    session     = ort.InferenceSession("models/emotion_model.onnx")
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    MODEL_OK    = True
except Exception as e:
    MODEL_OK    = False
    print(f"Emotion model load failed: {e}")

EMOTIONS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_DEFAULT = {
    "emotion":    "Neutral",
    "confidence": 0.0,
    "all_scores": {e:0.0 for e in EMOTIONS},
    "face_box":   None
}

def detect_emotion(frame):
    if not MODEL_OK:
        return _DEFAULT.copy()

    try:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30,30)
        )

        if len(faces) == 0:
            return _DEFAULT.copy()

        x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
        roi      = gray[y:y+h, x:x+w]
        roi      = cv2.resize(roi,(48,48)).astype("float32")/255.0
        roi      = np.expand_dims(np.expand_dims(roi,-1),0)

        preds    = session.run([output_name],{input_name:roi})[0][0]
        idx      = int(np.argmax(preds))
        conf     = float(round(preds[idx]*100,2))
        scores   = {EMOTIONS[i]:round(float(preds[i])*100,1)
                   for i in range(len(EMOTIONS))}

        return {
            "emotion":    EMOTIONS[idx],
            "confidence": conf,
            "all_scores": scores,
            "face_box":   (x,y,w,h)
        }
    except Exception:
        return _DEFAULT.copy()