import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import  DataLoader
import torch.nn as nn
import torch.nn.functional as F   
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


from PIL import Image
from Resnetmodeldiversified import CNN


import cv2
import csv

from pathlib import Path

#script for evaluation only

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder = Path(__file__).resolve().parent


model = CNN()
model.to(device)

emotion_labels = ["angry","disgust","fear","happy","sad","surprise"]


with open(current_folder / "output.csv", "w", newline="") as w:
            writer = csv.writer(w)
            for emotion in emotion_labels:
                writer.writerow([emotion])




state_dict = torch.load(
    current_folder / "Resnetbestweightslrweight0.001.pth",
    map_location=device
)


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


#

class Facefinder():
    def __call__(self, image):

        np_image = np.array(image)               
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image,1.1,5,minSize=(40,40))

        if len(faces) == 0:
            return image

        x,y,w,h = faces[0]
        face = np_image[y:y+h, x:x+w]

        face = Image.fromarray(cv2.cvtColor(face,cv2.COLOR_BGR2RGB))

        return face     




model.eval()

img_transforms = transforms.Compose([
    Facefinder(),
    transforms.Grayscale(1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


########################## for evaluation on other datasets change the root here #################################################################################

valdata = datasets.ImageFolder(
    root=current_folder / "test",
    transform=img_transforms
)

data_loader = DataLoader(dataset=valdata, batch_size=256, shuffle=False)

lossFunction = nn.CrossEntropyLoss()
model.load_state_dict(state_dict)

correct = 0
total = 0
eval_loss = 0.0


image_paths = [sample[0] for sample in valdata.samples]
image_index = 0


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
