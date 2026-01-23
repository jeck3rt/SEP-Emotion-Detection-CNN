#
import torch.nn as nn

#This is only the model so it can be loaded into demo.py

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
