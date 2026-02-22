import numpy as np
import time

import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2

from utils import (
    emotion_labels, device, img_transforms,
    face_classifier, load_model, draw_emotion_bars
)

# Script for the live demo including the Model, Grad-CAM, and a Webcam Loop

model = load_model()

# Initialize GradCAM once outside the loop for performance
gradlayer = [model.conv3_1_3, model.conv3_2_1, model.conv3_2_2]
cam = GradCAM(model=model, target_layers=gradlayer)

webcam = cv2.VideoCapture(0)


def face_detection_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


def emotion_detection(frame, faces):
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        pilimg = Image.fromarray(face)

        tensorimage = img_transforms(pilimg).unsqueeze(0).to(device)

        outputs = model(tensorimage)
        probs = torch.softmax(outputs, dim=1)
        probs_np = probs.detach().cpu().numpy()[0]

        predicted_class = np.argmax(probs_np)
        confidence = probs_np[predicted_class]

        # Show emotion label + confidence % above the face box
        label_text = f"{emotion_labels[predicted_class]}  {confidence * 100:.0f}%"
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Grad-CAM heatmap overlay on face region
        classes = [ClassifierOutputTarget(predicted_class)]
        heatmap = cam(input_tensor=tensorimage, targets=classes)
        heatmap = heatmap.squeeze(0) * 255
        heatmap = np.array(Image.fromarray(np.uint8(heatmap)).resize((w, h)))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.addWeighted(heatmap, 0.3, heatmap_bgr, 0.7, 0)
        frame[y:y + h, x:x + w] = heatmap

    return probs_np if len(faces) > 0 else None


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# Loop for Webcam demo
prev_time = time.time()

while True:
    result, video_frame = webcam.read()
    if result is False:
        break

    faces = face_detection_box(video_frame)
    probs = emotion_detection(video_frame, faces)

    if probs is not None:
        draw_emotion_bars(video_frame, probs)

    # FPS counter
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-9)
    prev_time = curr_time
    draw_fps(video_frame, fps)

    cv2.imshow("Model", video_frame)

    # Press 'q' to quit, or click the window's X button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if cv2.getWindowProperty("Model", cv2.WND_PROP_VISIBLE) < 1:
        break

webcam.release()
cv2.destroyAllWindows()
