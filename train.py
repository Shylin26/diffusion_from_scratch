import torch
import torch.nn.functional as F
from torch.optim import Adam
from forward import Diffusion
from reverse_process import NoisePredictor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

dataset=MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

device="cuda" if torch.cuda.is_available() else "cpu"
loader=DataLoader(dataset,batch_size=32,shuffle=True)
model=NoisePredictor().to(device)
diffusion=Diffusion(device=device)
optimizer=Adam(model.parameters(),lr=1e-4)

epochs=5
T=300
for epoch in range(epochs):
    for x0, _ in loader:
        x0=x0.to(device)
    # 1. sample random timesteps
        t=torch.randint(0,T,(x0.size(0),),device=device)
    #2. generate noisy images
        xt,noise=diffusion.forward(x0,t)
    # predict noise
        noise_pred=model(xt,t)
    # compute loss
        loss=F.mse_loss(noise_pred,noise)
    # 5. optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch{epoch}|Loss:{loss.item():.4f}")
torch.save(model.state_dict(),"model.pth")


