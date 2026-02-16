#
import torch.nn as nn

num_labels = 6
#This is only the model so it can be loaded into demo.py

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
