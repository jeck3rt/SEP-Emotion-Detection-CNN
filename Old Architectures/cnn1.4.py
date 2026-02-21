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


from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


##emotion_labels = ["surprised","fearful","disgusted","happy","sad","angry"]


##parameters
batch_size = 100
epochs = 100
num_labels = 6
input_size = 64*64


## this needs somekind of interpolation to lazy rn ngl ts pmo

train_transform = transforms.Compose([
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size =2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,kernel_size=5,padding=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,padding=2,stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=64*8*8,out_features=120) 
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,6)
        self.relu = torch.nn.ReLU()
         


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x) 
        
        return x
    
#checks if GPU is available to use
device = "cuda" if torch.cuda.is_available() else "cpu"

myCNN = CNN().to(device)

#DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
data_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

#Loss and Optimizer Functions
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(myCNN.parameters(), lr=0.001)



#Training Loop
myCNN.train()

for epoch in range(epochs):
    for step, (data, targets) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        scores = myCNN(data)
        loss = lossFunction(scores, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    
    accuracy= []
    


torch.save(
    myCNN.state_dict(),
    "C:\\Users\\ecker\\Desktop\\Fer2013\\cnn_weights3.pth"
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
        loss = lossFunction(outputs, targets)

        eval_loss += loss.item()
        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()
        total += targets.size(0)

avg_loss = eval_loss / len(data_loader)
accuracy = correct / total

print(avg_loss)
print(accuracy)

