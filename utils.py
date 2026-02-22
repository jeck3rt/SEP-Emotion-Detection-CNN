# Shared utilities for demo.py, videodemo.py, ImageExampleScripts.py, and evaluation.py

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2

from cnn import CNN


emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder = Path(__file__).resolve().parent

# ── Face classifier ───────────────────────────────────────────────────────────

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class Facefinder:
    """Crop the first detected face from a PIL image (used during dataset eval)."""
    def __call__(self, image):
        np_image = np.array(image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return image
        x, y, w, h = faces[0]
        face = np_image[y:y + h, x:x + w]
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))


# ── Transforms ────────────────────────────────────────────────────────────────

# For demo / videodemo / ImageExampleScripts (face already cropped by OpenCV)
img_transforms = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# For evaluation on dataset folders (includes Facefinder pre-crop)
eval_transforms = transforms.Compose([
    Facefinder(),
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    """Load the CNN with saved weights and set to eval mode."""
    model = CNN()
    model.to(device)
    state_dict = torch.load(current_folder / "weights.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Visualization helpers ─────────────────────────────────────────────────────

def draw_emotion_bars(frame, probs):
    """Draw a semi-transparent probability bar panel in the bottom-right corner."""
    panel_w = 220
    panel_h = 20 + len(emotion_labels) * 30
    margin = 10

    h, w = frame.shape[:2]
    x_start = w - panel_w - margin
    y_start = h - panel_h - margin

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x_start, y_start),
                  (x_start + panel_w, y_start + panel_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, prob) in enumerate(zip(emotion_labels, probs)):
        y = y_start + 15 + i * 30
        bar_x = x_start + 90
        bar_max_w = 110
        bar_h = 14
        color = (0, 255, 0) if i == np.argmax(probs) else (200, 120, 50)

        cv2.putText(frame, f"{label}:", (x_start + 5, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + bar_h), (80, 80, 80), -1)
        filled_w = int(prob * bar_max_w)
        cv2.rectangle(frame, (bar_x, y), (bar_x + filled_w, y + bar_h), color, -1)
        cv2.putText(frame, f"{prob * 100:.0f}%", (bar_x + bar_max_w + 3, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
