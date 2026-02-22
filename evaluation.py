import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

import cv2
import csv
from pathlib import Path
from collections import defaultdict

from utils import (
    emotion_labels, device, eval_transforms,
    current_folder, load_model
)

#################################################################################### script for evaluation only #############################################################################

model = load_model()

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

########################## for evaluation on other datasets change the root here #################################################################################

valdata = datasets.ImageFolder(
    root=current_folder / "test",
    transform=eval_transforms
)

# The dataset may contain extra classes not known to the model (e.g. "neutral" in FERPlus).
# Filter samples to only the 6 trained emotions and remap labels to 0-5.
valid_class_indices = {valdata.class_to_idx[c] for c in emotion_labels if c in valdata.class_to_idx}
label_remap = {valdata.class_to_idx[c]: i for i, c in enumerate(emotion_labels) if c in valdata.class_to_idx}
filtered_samples = [
    (path, label_remap[label])
    for path, label in valdata.samples
    if label in valid_class_indices
]
valdata.samples = filtered_samples
valdata.targets = [label for _, label in filtered_samples]

print(f"Classes found in dataset: {list(valdata.class_to_idx.keys())}")
print(f"Evaluating on {len(valdata)} images across {len(valid_class_indices)} classes.\n")

data_loader = DataLoader(dataset=valdata, batch_size=256, shuffle=False)

lossFunction = nn.CrossEntropyLoss()

correct = 0
total = 0
eval_loss = 0.0

image_paths = [sample[0] for sample in valdata.samples]
image_index = 0

# For confusion matrix and per-class visualization
all_preds = []
all_targets = []

# Collect up to N sample images per class for the image grid
SAMPLES_PER_CLASS = 5
collected = defaultdict(list)   # true_label -> list of (img_tensor, pred_label)


with open(current_folder / "output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "img",
        "angryoutput",
        "disgustoutput",
        "fearoutput",
        "happyoutput",
        "sadoutput",
        "surpriseoutput"
    ])

with torch.no_grad():
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)
        loss = lossFunction(outputs, targets)

        eval_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

        probs = F.softmax(outputs, dim=1)
        probs_np = probs.cpu().numpy()

        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        # Collect sample images per class for later grid visualization
        for i in range(len(targets)):
            true_label = targets[i].item()
            pred_label = predictions[i].item()
            if len(collected[true_label]) < SAMPLES_PER_CLASS:
                collected[true_label].append((data[i].cpu(), pred_label))

        # Append results to CSV
        with open(current_folder / "output.csv", "a", newline="") as f:
            writer = csv.writer(f)
            for prob in probs_np:
                img_name = Path(image_paths[image_index]).name
                writer.writerow([img_name] + prob.tolist())
                image_index += 1


avg_loss = eval_loss / len(data_loader)
accuracy = correct / total

print("Loss:", avg_loss)
print("Accuracy:", accuracy)


# =====================================================================
# VISUALIZATION
# =====================================================================

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)
num_classes = len(emotion_labels)

# ── 1. Confusion Matrix + Per-class Accuracy ─────────────────────────

conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_targets, all_preds):
    conf_matrix[t][p] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix heatmap
ax_cm = axes[0]
im = ax_cm.imshow(conf_matrix, cmap="Blues")
ax_cm.set_xticks(range(num_classes))
ax_cm.set_yticks(range(num_classes))
ax_cm.set_xticklabels(emotion_labels, rotation=45, ha="right")
ax_cm.set_yticklabels(emotion_labels)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
ax_cm.set_title(f"Confusion Matrix\nOverall Accuracy: {accuracy:.1%}")
plt.colorbar(im, ax=ax_cm)
for i in range(num_classes):
    for j in range(num_classes):
        text_color = "white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black"
        ax_cm.text(j, i, str(conf_matrix[i, j]),
                   ha="center", va="center", color=text_color, fontsize=9)

# Per-class accuracy bar chart
ax_bar = axes[1]
class_acc = []
for i in range(num_classes):
    mask = all_targets == i
    class_acc.append(float((all_preds[mask] == i).mean()) if mask.sum() > 0 else 0.0)

colors = ["#2196F3" if a >= accuracy else "#FF5722" for a in class_acc]
bars = ax_bar.bar(emotion_labels, class_acc, color=colors)
ax_bar.set_ylim(0, 1.05)
ax_bar.axhline(y=accuracy, color="gray", linestyle="--",
               label=f"Overall: {accuracy:.1%}")
ax_bar.set_xlabel("Emotion")
ax_bar.set_ylabel("Accuracy")
ax_bar.set_title("Per-class Accuracy\n(Blue ≥ overall, Orange < overall)")
ax_bar.legend()
for bar, acc in zip(bars, class_acc):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(current_folder / "evaluation_stats.png", dpi=150)
plt.show()
print("Saved: evaluation_stats.png")


# ── 2. Sample Image Grid: actual face images with predicted labels ────

fig2, axes2 = plt.subplots(
    num_classes, SAMPLES_PER_CLASS,
    figsize=(SAMPLES_PER_CLASS * 2.2, num_classes * 2.4)
)
fig2.suptitle(
    "Sample Predictions  |  Green title = correct,  Red title = wrong",
    fontsize=12, y=1.01
)

for row, true_label in enumerate(range(num_classes)):
    samples = collected[true_label]
    for col in range(SAMPLES_PER_CLASS):
        ax = axes2[row][col]
        if col < len(samples):
            img_tensor, pred_label = samples[col]
            # Denormalize: pixel = tensor * 0.5 + 0.5
            img_np = img_tensor.squeeze(0).numpy() * 0.5 + 0.5
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np, cmap="gray")
            is_correct = (pred_label == true_label)
            color = "green" if is_correct else "red"
            ax.set_title(f"Pred: {emotion_labels[pred_label]}",
                         fontsize=7, color=color, pad=2)
            if col == 0:
                ax.set_ylabel(f"True:\n{emotion_labels[true_label]}",
                              fontsize=7, labelpad=4)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(current_folder / "evaluation_samples.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: evaluation_samples.png")
