import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from model import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
multi_gpu = torch.cuda.device_count() > 1

transform = transforms.Compose([transforms.Scale((24,14)),
	transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor()])

dset = datasets.ImageFolder(root='train', transform=transform)
dloader = torch.utils.data.DataLoader(dset,
	batch_size=8, shuffle=True, num_workers=12)

net = nn.DataParallel(Net()) if multi_gpu else Net()

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):
	running_loss = 0.0
	for i, (inp, lab) in enumerate(dloader, 0):
		inp = inp.to(device)
		lab = lab.to(device)
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
torch.save(net.module if multi_gpu else net, 'model.pt')
