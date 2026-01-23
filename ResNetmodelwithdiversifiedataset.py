import numpy as np


import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm


from PIL import Image
import cv2

from pathlib import Path


##emotion_labels = ["surprised","fearful","disgusted","happy","sad","angry"]


current_folder = Path(__file__).resolve().parent
##parameters
batch_size = 64
epochs = 75
num_labels = 6 
input_size = 64*64


#Loads haarcascade face classifier 
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



#The dataset is mostly made up of Fer2013, however we added more data from another Kaggle dataset. The new Data has a different size and pictures where not necessarily centered, so this class is used to later transform  the images so they are.
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


#transforms for data
train_transform = transforms.Compose([
    Facefinder(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
     transforms.Grayscale(1),
     transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize(0.5,0.5)

])

test_transform = transforms.Compose([
     Facefinder(),
     transforms.Grayscale(1),
     transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize(0.5,0.5)
])


#data and loaders
train_data = datasets.ImageFolder(
    root=current_folder / "traindiv",
    transform=train_transform
)

test_data = datasets.ImageFolder(
    root=current_folder / "testdiv",
    transform=test_transform
)





#ResNet-esque model with 4 skip connections

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
         

        self.relu = nn.ReLU()

        self.conv1= nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2, stride=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
                    
        
        self.conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batchnorm2_1_1 =  nn.BatchNorm2d(64)
        self.conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batchnorm2_1_2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.5)
                                    
        
        self.conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batchnorm2_2_1 = nn.BatchNorm2d(64)
        self.conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.batchnorm2_2_2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.5)
                                    
        
        self.conv3_1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.batchnorm3_1_1 = nn.BatchNorm2d(128)
        self.conv3_1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.batchnorm3_1_2 = nn.BatchNorm2d(128)

        self.conv3_1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=0, stride=2)
        self.dropout3 =  nn.Dropout(0.5)
                                    
        
        self.conv3_2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.batchnorm3_2_1 = nn.BatchNorm2d(128)
        self.conv3_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.batchnorm3_2_2 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.5)
                                    

        self.finalblock = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(),
                                        nn.Linear(in_features=128, out_features=512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=512, out_features=6)

        )

    def forward(self, x):
               
        x = self.relu(self.batchnorm1(self.conv1(x)))
        op1 = self.maxpool1(x)
        
        
        
        x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(op1)))    
        x = self.batchnorm2_1_2(self.conv2_1_2(x))                 
        x = self.dropout1(x)
    
        op2_1 = self.relu(x + op1)
 
        x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(op2_1)))  
        x = self.batchnorm2_2_2(self.conv2_2_2(x))                
        x = self.dropout2(x)
        op2 = self.relu(x + op2_1)
    
        
    
        x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(op2)))    
        x = self.batchnorm3_1_2(self.conv3_1_2(x))                 
        x = self.dropout3(x)
        
        op2 = self.conv3_1_3(op2) 
        
        op3_1 = self.relu(x + op2)
        
        x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(op3_1)))  
        x = self.batchnorm3_2_2(self.conv3_2_2(x))                  
        x = self.dropout4(x)
        
        op3 = self.relu(x + op3_1)

        


        x = self.finalblock(op3)
        
        return x
    
#checks if GPU is available to use
device = "cuda" if torch.cuda.is_available() else "cpu"

myCNN = CNN().to(device)

#DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
data_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

#Loss and Optimizer Functions
evalrawweights = [6.18, 20.39, 6.05, 3.55, 5.36, 6.42]
meanrawweight = sum(evalrawweights) / len(evalrawweights)
evalweights = np.array(evalrawweights) / meanrawweight
evalweights = torch.tensor(evalweights, dtype=torch.float32).to(device)

trainrawweights = [6.27, 12.35, 6.24, 3.75, 5.93, 6.10]
meanevalweight = sum(trainrawweights) / len(trainrawweights)
trainweights = np.array(trainrawweights) / meanevalweight
trainweights = torch.tensor(trainweights, dtype=torch.float32).to(device)

evalLoss = nn.CrossEntropyLoss(weight= evalweights)
trainingloss = nn.CrossEntropyLoss(weight= trainweights)
optimizer = optim.Adam(myCNN.parameters(), lr=0.0001)



#Training Loop
myCNN.train()

for epoch in range(epochs):
    for step, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        scores = myCNN(data)
        loss = trainingloss(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    accuracy= []
    


torch.save(
    myCNN.state_dict(),
    current_folder / "ResNetdiversified.pth"
)



#eval loop
myCNN.eval()  

correct = 0
total = 0
eval_loss = 0.0

with torch.no_grad():  
    for data, targets in data_loader: 
        data = data.to(device)
        targets = targets.to(device)

        outputs = myCNN(data)
        loss = evalLoss(outputs, targets)

        eval_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

avg_loss = eval_loss / len(data_loader)
accuracy = correct / total

print(avg_loss)
print(accuracy)
