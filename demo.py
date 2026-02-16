import numpy as np

import torch
from torchvision import transforms



from PIL import Image
from cnn import CNN

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import cv2
import csv
from pathlib import Path



#Script for the live demo including the Model, Grad-CAM, and a Webcam Loop 



emotion_labels = ["angry","disgust","fear","happy","sad","surprise"]
#checks if cnn should run on gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder = Path(__file__).resolve().parent

model = CNN()
model.to(device)





state_dict = torch.load(
    current_folder / "weights.pth",
    map_location=device
)




img_transforms = transforms.Compose(
    [
    transforms.Grayscale(1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])


model.load_state_dict(state_dict)

model.eval()


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



webcam = cv2.VideoCapture(0)


def face_detection_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image,1.1,5,minSize=(40,40))
    for(x,y,w,h) in faces:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),4)
    return faces


def emotion_detection(frame,faces):
    for(x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) 
        pilimg = Image.fromarray(face)


        

        tensorimage = img_transforms(pilimg).unsqueeze(0).to(device)
    
        #with torch.no_grad():       
        outputs = model(tensorimage)
        probs = torch.softmax(outputs, dim=1)
        probs_np = probs.detach().cpu().numpy()[0]

        predicted_class = np.argmax(probs_np)

        cv2.putText(frame, str(predicted_class), (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        #initialize Grad-CAM
        classes = [ClassifierOutputTarget(predicted_class)] 
        gradlayer = [model.conv3_1_3, model.conv3_2_1, model.conv3_2_2]

        cam = GradCAM(model=model, target_layers=gradlayer) 
        heatmap = cam(input_tensor=tensorimage,targets=classes) 
        heatmap = heatmap.squeeze(0) 
        heatmap = heatmap*255 
        heatmap = Image.fromarray(np.uint8(heatmap)) 
        heatmap = heatmap.resize((w, h)) 
        heatmap= np.array(heatmap) 
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) 
        heatmap_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR) 
        heatmap = cv2.addWeighted(heatmap, 0.3, heatmap_bgr, 1 - 0.3, 0) 
        frame[y:y+h, x:x+w] = heatmap 

    
      



#Loop for Webcam demo
while True:
    result, video_frame = webcam.read()
    if result is False:
        break

    faces = face_detection_box(
        video_frame
    )

    prediction = emotion_detection(
        video_frame, faces
    )

    cv2.imshow(
        "Model", video_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()







