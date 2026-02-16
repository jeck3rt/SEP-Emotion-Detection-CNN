import numpy as np



import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt


from torch.optim.lr_scheduler import ReduceLROnPlateau

from PIL import Image
import cv2

from pathlib import Path

from collections import Counter

##emotion_labels = ["surprised","fearful","disgusted","happy","sad","angry"]

if __name__ == "__main__":

    current_folder = Path(__file__).resolve().parent
    torch.multiprocessing.freeze_support()

    ##parameters
    batch_size = 64
    epochs = 350
    num_labels = 6 
    input_size = 64*64


    #Loads haarcascade face classifier 
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )



    #for our live demo we used the haarclassifier to find faces in the webcam.
    #to give our cnn more consistency I decided to add the function to image pre-processing so the cnn would see similiar inputs the whole time.
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


    face_transform = transforms.Compose([
        Facefinder()   # ONLY face detection here
    ])


    #precropping the images to save computing time
    def precrop_dataset(src_root, dst_root):
        dst_root.mkdir(parents=True, exist_ok=True)

        dataset = datasets.ImageFolder(root=src_root, transform=face_transform)

        for img_path, label in dataset.samples:
            class_name = dataset.classes[label]

            img = Image.open(img_path).convert("RGB")
            cropped = face_transform(img)

            out_dir = dst_root / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / Path(img_path).name
            cropped.save(out_path)

    if not (current_folder/ "train_cropped").is_dir():
        precrop_dataset((current_folder / "train"), (current_folder / "train_cropped"))

    if not (current_folder/ "test_cropped").is_dir():
        precrop_dataset((current_folder / "test"), (current_folder / "test_cropped"))


    if not (current_folder/ "validation_cropped").is_dir():
        precrop_dataset((current_folder / "validation"), (current_folder / "validation_cropped"))

    #Image pre-processing 
    train_transform = transforms.Compose([
        transforms.ColorJitter(0.1,0.1),
        transforms.Grayscale(1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value="random"),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    #used for validation and test data

    test_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    #data and loaders
    train_data = datasets.ImageFolder(root=current_folder / "train_cropped", transform = train_transform)
    test_data = datasets.ImageFolder(root=current_folder / "test_cropped", transform = test_transform)
    eval_data = datasets.ImageFolder(root=current_folder / "validation_cropped", transform = test_transform)



    #ResNet-esque model with 6 skip connections

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.dropout = nn.Dropout(0.4)
            self.relu = nn.ReLU()


            self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=1)
            self.batchnorm1 = nn.BatchNorm2d(64)
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


            self.conv2_1_1 = nn.Conv2d(64, 64, 3, padding=1)
            self.batchnorm2_1_1 = nn.BatchNorm2d(64)
            self.conv2_1_2 = nn.Conv2d(64, 64, 3, padding=1)
            self.batchnorm2_1_2 = nn.BatchNorm2d(64)

            self.conv2_2_1 = nn.Conv2d(64, 64, 3, padding=1)
            self.batchnorm2_2_1 = nn.BatchNorm2d(64)
            self.conv2_2_2 = nn.Conv2d(64, 64, 3, padding=1)
            self.batchnorm2_2_2 = nn.BatchNorm2d(64)

            self.conv3_1_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
            self.batchnorm3_1_1 = nn.BatchNorm2d(128)
            self.conv3_1_2 = nn.Conv2d(128, 128, 3, padding=1)
            self.batchnorm3_1_2 = nn.BatchNorm2d(128)
            self.conv3_1_3 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

            self.conv3_2_1 = nn.Conv2d(128, 128, 3, padding=1)
            self.batchnorm3_2_1 = nn.BatchNorm2d(128)
            self.conv3_2_2 = nn.Conv2d(128, 128, 3, padding=1)
            self.batchnorm3_2_2 = nn.BatchNorm2d(128)


            self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
            self.batchnorm4_1 = nn.BatchNorm2d(256)
            self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
            self.batchnorm4_2 = nn.BatchNorm2d(256)
            self.conv4_3 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        
            self.conv4_4_1 = nn.Conv2d(256, 256, 3, padding=1)
            self.batchnorm4_4_1 = nn.BatchNorm2d(256)
            self.conv4_4_2 = nn.Conv2d(256, 256, 3, padding=1)
            self.batchnorm4_4_2 = nn.BatchNorm2d(256)

            


        
            self.finalblock = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(512, num_labels)
            )

        def forward(self, x):

            x = self.relu(self.batchnorm1(self.conv1(x)))
            x = self.maxpool1(x)

           
            skip_1 = x
            x = self.relu(self.batchnorm2_1_1(self.conv2_1_1(x)))
            x = self.batchnorm2_1_2(self.conv2_1_2(x))
            x = self.dropout(self.relu(x + skip_1))

          
            skip_2 = x
            x = self.relu(self.batchnorm2_2_1(self.conv2_2_1(x)))
            x = self.batchnorm2_2_2(self.conv2_2_2(x))
            x = self.dropout(self.relu(x + skip_2))

           
            skip_3 = self.conv3_1_3(x)
            x = self.relu(self.batchnorm3_1_1(self.conv3_1_1(x)))
            x = self.batchnorm3_1_2(self.conv3_1_2(x))
            x = self.dropout(self.relu(x + skip_3))

            
            skip_4 = x
            x = self.relu(self.batchnorm3_2_1(self.conv3_2_1(x)))
            x = self.batchnorm3_2_2(self.conv3_2_2(x))
            x = self.dropout(self.relu(x + skip_4))

            
            skip_5 = self.conv4_3(x)
            x = self.relu(self.batchnorm4_1(self.conv4_1(x)))
            x = self.batchnorm4_2(self.conv4_2(x))
            x = self.dropout(self.relu(x + skip_5))

            
            skip_6 = x
            x = self.relu(self.batchnorm4_4_1(self.conv4_4_1(x)))
            x = self.batchnorm4_4_2(self.conv4_4_2(x))
            x = self.dropout(self.relu(x + skip_6))

            x = self.finalblock(x)
            return x
        

    #checks if GPU is available to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    myCNN = CNN().to(device)

    #DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True, num_workers=4)
    data_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False, num_workers=4)
    eval_loader = DataLoader(dataset=eval_data, batch_size=batch_size,shuffle=False, num_workers=4)



    #Loss and Optimizer Functions

    train_targets = train_data.targets
    train_counter = Counter(train_targets)
    class_counts_train = [train_counter[i] for i in range(len(train_data.classes))]

    test_targets = test_data.targets
    test_counter = Counter(test_targets)
    class_counts_test = [test_counter[i] for i in range(len(test_data.classes))]



    evalrawweights = [sum(class_counts_test)/c for c in class_counts_test]
    meanrawweight = sum(evalrawweights) / len(evalrawweights)
    evalweights = np.array(evalrawweights) / meanrawweight
    evalweights = torch.tensor(evalweights, dtype=torch.float32).to(device)

    trainrawweights = [sum(class_counts_train)/c for c in class_counts_train]
    meanevalweight = sum(trainrawweights) / len(trainrawweights)
    trainweights = np.array(trainrawweights) / meanevalweight
    trainweights = torch.tensor(trainweights, dtype=torch.float32).to(device)

    trainingloss = nn.CrossEntropyLoss(weight=trainweights)
    evalLoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(myCNN.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=3, threshold=0.001, threshold_mode='abs')



    # variables used in the training/eval loop for early stopping and performance evaluation

    best_loss = 100 # This is just a high starting number so the first loss value is the the actual best 
    grace_period  = 0


    train_losses = []
    val_losses = []
    val_accuracies = []  

    #Training Loop
    scaler = torch.amp.GradScaler()
    for epoch in range(epochs):
        myCNN.train()
        
        epoch_train_loss = 0.0
        
        for step, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device):
                scores = myCNN(data)
                loss = trainingloss(scores, targets)


            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #loss.backward()
            #optimizer.step()

            epoch_train_loss += loss.item()


        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        accuracy= []
        


        #eval loop
        myCNN.eval()

        correct = 0
        total = 0
        eval_loss = 0.0

        with torch.no_grad():  
            for data, targets in eval_loader: 
                data = data.to(device)
                targets = targets.to(device)

                outputs = myCNN(data)
                loss = evalLoss(outputs, targets)

                eval_loss += loss.item()
                predictions = outputs.argmax(dim=1)

                correct += (predictions == targets).sum().item()
                total += targets.size(0)


        avg_loss = eval_loss / len(eval_loader)
        val_losses.append(avg_loss)


        accuracy = correct / total
        val_accuracies.append(accuracy)



        print(f"Average loss:{avg_loss},accuracy of current weights:{accuracy}, current epoch:{epoch}")

        scheduler.step(avg_loss)


        #early stopping
        if(epoch >=20):
            if(avg_loss <= best_loss ):
        
                best_loss = avg_loss
                torch.save(
                myCNN.state_dict(),
                current_folder / "ResNebut100epochsmin.pth"
                ) 
                grace_period = 0
            elif(avg_loss > best_loss ):
                grace_period +=1
                if(grace_period >=10):
                    print("Overfitting stopping now zzzZZZzzz")
                    break



    #plots our epochs performance
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


