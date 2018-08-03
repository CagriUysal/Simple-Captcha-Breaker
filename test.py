import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from model import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Scale((24,14)), transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor()])
dset = datasets.ImageFolder(root='train', transform=transform)
dloader = torch.utils.data.DataLoader(dset,
	batch_size=8, shuffle=True, num_workers=12)

net = torch.load('model.pt').to(device)
with torch.no_grad():
	for i, (inp,lab) in enumerate(dloader,0):
		ip1 = inp[0,:,:,:].numpy().transpose(1,2,0)
		ip2 = inp[1,:,:,:].numpy().transpose(1,2,0)
		ip3 = inp[2,:,:,:].numpy().transpose(1,2,0)
		ip4 = inp[3,:,:,:].numpy().transpose(1,2,0)
		ip5 = inp[4,:,:,:].numpy().transpose(1,2,0)
		ip6 = inp[5,:,:,:].numpy().transpose(1,2,0)
		ip7 = inp[6,:,:,:].numpy().transpose(1,2,0)
		ip8 = inp[7,:,:,:].numpy().transpose(1,2,0)

		inp = inp.to(device)
		lab = lab.to(device)

		outs = net(inp)
		_, pred = torch.max(outs,1)
		print(pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item(), pred[4].item(), pred[5].item(), pred[6].item(), pred[7].item())
		cv2.imshow('img1', ip1)
		cv2.imshow('img2', ip2)
		cv2.imshow('img3', ip3)
		cv2.imshow('img4', ip4)
		cv2.imshow('img5', ip5)
		cv2.imshow('img6', ip6)
		cv2.imshow('img7', ip7)
		cv2.imshow('img8', ip8)
		cv2.waitKey(0)
	
