import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

transform = transforms.Compose([transforms.Scale((24,14)), transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor()])
dset = datasets.ImageFolder(root='train', transform=transform)
dloader = torch.utils.data.DataLoader(dset,
	batch_size=4, shuffle=True, num_workers=1)


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 3, 3)
		#self.pool = nn.MaxPool2d(2,2)
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
net = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
	running_loss = 0.0
	for i, (inp, lab) in enumerate(dloader, 0):
		inp, lab = inp.cuda(), lab.cuda()
		optimizer.zero_grad()

		outs = net(inp)
		loss = criterion(outs, lab)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 100 == 99:
			print('[%d, %d] loss: %f' % (epoch + 1, i+1, running_loss))
			running_loss = 0.0
print('finished')
with torch.no_grad():
	for i, (inp,lab) in enumerate(dloader,0):
		ip = inp[0,:,:,:].numpy().transpose(1,2,0)
		inp, lab = inp.cuda(), lab.cuda()

		outs = net(inp)
		_, pred = torch.max(outs,1)
		print(pred[0])
		cv2.imshow('img', ip)
		cv2.waitKey(0)
	
