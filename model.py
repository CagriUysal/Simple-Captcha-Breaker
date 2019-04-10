import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 24x14x1 image tensor)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        # convolutional layer (sees 22x12x3 tensor)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3)
        # linear layer (20 * 10 * 3 -> 30)
        self.fc1 = nn.Linear(20*10*3, 30)
        # linear layer (30 -> 10)
        self.fc2 = nn.Linear(30, 10)
    def forward(self, x):
        # sequance of convolutional layers with relu activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten the image input
        x = x.view(-1, 20*10*3)
        # 1st hidden layer with relu activation
        x = F.relu(self.fc1(x))
        # output-layer
        x = self.fc2(x)
        return x
