import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 3, 3)
		self.conv2 = nn.Conv2d(3, 3, 3)
		self.fc1 = nn.Linear(200*3, 30)
		self.fc2 = nn.Linear(30, 10)
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = x.view(-1, 200*3)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
