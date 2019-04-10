import torch
from torchvision import transforms, datasets
from model import Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((24,14)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()])

dset = datasets.ImageFolder(root='test', transform=transform)
dloader = torch.utils.data.DataLoader(dset,
    batch_size=8, shuffle=True, num_workers=12)

total = 0
correct = 0

net = torch.load('model.pt').to(device)
with torch.no_grad():
    for i, (inp,lab) in enumerate(dloader,0):
        inp = inp.to(device)
        lab = lab.to(device)

        # forward pass
        outs = net(inp)
        # convert output scores to predicted class
        _, pred = torch.max(outs,1)
        correct += (pred == lab).sum().item()
        total += lab.size(0)

print('Accuracy: %f %%' % (100*correct/total))
