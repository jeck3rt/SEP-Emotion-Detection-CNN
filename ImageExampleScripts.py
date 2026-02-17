import numpy as np

import torch
from torchvision import transforms



from PIL import Image
from cnn import CNN

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


import cv2

from pathlib import Path



#Script for the final report paper to show gradcam on example pictures #######################################



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





def face_detection_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image,1.1,5,minSize=(40,40))
    for(x,y,w,h) in faces:
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),4)
    return faces


def emotion_detection(frame):
    faces = face_detection_box(frame)
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

    return frame

    
img_path = current_folder / "test/happy/happy3.png"    
img = Image.open(img_path).convert("RGB")
img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
gradcamresults = emotion_detection(img_bgr)

output = current_folder / "outputimage.png"

cv2.imwrite(str(output), gradcamresults)



