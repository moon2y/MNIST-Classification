
import torch.nn as nn

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):

        # write your codes here
        super(LeNet5,self).__init__()
        self.c1 = nn.Conv2d(1,6,5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(6,16,5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(16,120,5)
        self.n1 = nn.Linear(120,84)
        self.relu = nn.ReLU()
        self.n2 = nn.Linear(84,10)

    def forward(self, img):

        # write your codes here
        x = self.c1(img)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.c2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.c3(x)
        x = self.relu(x)
        x = x.view(-1, 120)
        #x = nn.Flatten(x)
        x = self.n1(x)
        x = self.relu(x)
        x = self.n2(x)

        return x

class LeNet5_regularized(nn.Module):
    """ LeNet-5 with Dropout and Batch Normalization """
    def __init__(self):
        super(LeNet5_regularized, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.maxpool1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(16, 120, 5)
        self.bn3 = nn.BatchNorm2d(120)
        self.n1 = nn.Linear(120, 84)
        self.dropout1 = nn.Dropout(0.5)
        self.n2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()
        
    def forward(self, img):
        x = self.c1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.view(-1, 120)
        #x = nn.Flatten(x)
        x = self.n1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.n2(x)
        
        return x

class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self, input_size):

        # write your codes here
        super(CustomMLP,self).__init__()
        self.n1 = nn.Linear(input_size,60)
        self.relu = nn.ReLU()
        self.n2 = nn.Linear(60,20)
        self.n3 = nn.Linear(20,10)

    def forward(self, img):

        # write your codes here
        x = self.n1(img)
        x = self.relu(x)
        x = self.n2(x)
        x = self.relu(x)
        x = self.n3(x)
        x = x.squeeze(dim=1)

        return x
