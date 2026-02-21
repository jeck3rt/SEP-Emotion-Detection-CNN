import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F   
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch import optim
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


##emotion_labels = ["surprised","fearful","disgusted","happy","sad","angry"]


##parameters
batch_size = 64
epochs = 75
num_labels = 6 
input_size = 64*64


## this needs somekind of interpolation to lazy rn ngl ts pmo

train_transform = transforms.Compose([
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
     transforms.Grayscale(1),
     transforms.Resize((64,64)),
     transforms.ToTensor(),
     transforms.Normalize(0.5,0.5)
])


#data and loaders
train_data = datasets.ImageFolder(
    root=r"C:\Users\ecker\Desktop\Fer2013\train",
    transform=train_transform
)

test_data = datasets.ImageFolder(
    root=r"C:\Users\ecker\Desktop\Fer2013\test",
    transform=test_transform
)






class CNN(nn.Module):
    def __init__(self):
        super().__init__()
         
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, stride=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU()
                                    )
        
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU()
                                    )
        
        self.layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
    
        self.fc1 = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(in_features=64*16*16, out_features=120),
                                nn.ReLU()
        )


        self.fc2 = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(in_features=120, out_features=84),
                                nn.ReLU()
        )

        self.fc3 = nn.Sequential(
                                nn.Linear(in_features=84, out_features=6),
                                
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x) 
        
        return x
    
#checks if GPU is available to use
device = "cuda" if torch.cuda.is_available() else "cpu"

myCNN = CNN().to(device)

#DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
data_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

#Loss and Optimizer Functions
evalrawweights = [6.2, 53.55, 5.8, 3.35, 4.77, 7.15]
meanrawweight = sum(evalrawweights) / len(evalrawweights)
evalweights = np.array(evalrawweights) / meanrawweight
evalweights = torch.tensor(evalweights, dtype=torch.float32).to(device)

trainrawweights = [5.94, 53.3, 5.8, 3.29, 4.92, 7.50]
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
    "C:\\Users\\ecker\\Desktop\\Fer2013\\cnn_weightslroutput32first2.pth"
)



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
