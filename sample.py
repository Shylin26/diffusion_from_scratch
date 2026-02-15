import torch
import matplotlib.pyplot as plt
from forward import Diffusion
from reverse_process import NoisePredictor
from torchvision.utils import save_image
device="cuda" if torch.cuda.is_available()else "cpu"
#load model
model=NoisePredictor().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()
diffusion=Diffusion(device=device)
xt=torch.randn(1,1,28,28).to(device)
# start from pure noise
with torch.no_grad():
    for t in reversed(range(diffusion.T)):
        t_tensor=torch.tensor([t],device=device)
        xt=diffusion.sample_step(model,xt,t_tensor)
img =xt.clone().detach().cpu().squeeze()
img=(img.clamp(-1,1)+1)/2
save_image(img,"sample.png")
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.show()
print("Model loaded successfully")