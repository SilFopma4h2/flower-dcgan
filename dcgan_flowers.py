!pip install -q kagglehub

import kagglehub, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = kagglehub.dataset_download("alxmamaev/flowers-recognition")
root = os.path.join(path, "flowers")
if not os.path.isdir(root) or not os.path.isdir(os.path.join(root, os.listdir(root)[0])):
    root = path

image_size = 64
batch_size = 128
nz, ngf, ndf, nc = 100, 64, 64, 3
num_epochs = 150

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(root=root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1)

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
os.makedirs("samples", exist_ok=True)

for epoch in range(num_epochs):
    for i, (real, _) in enumerate(dataloader):
        real = real.to(device)
        b_size = real.size(0)
        label_real = torch.full((b_size,), 1.0, device=device)
        label_fake = torch.full((b_size,), 0.0, device=device)

        netD.zero_grad()
        out_real = netD(real)
        loss_real = criterion(out_real, label_real)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        out_fake = netD(fake.detach())
        loss_fake = criterion(out_fake, label_fake)

        lossD = loss_real + loss_fake
        lossD.backward()
        optD.step()

        netG.zero_grad()
        out = netD(fake)
        lossG = criterion(out, label_real)
        lossG.backward()
        optG.step()

        if i % 100 == 0:
            print(f"epoch {epoch} step {i} lossD {lossD.item():.4f} lossG {lossG.item():.4f}")

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    utils.save_image(fake, f"samples/epoch_{epoch}.png", normalize=True)

display(Image.open(f"samples/epoch_{num_epochs-1}.png"))
