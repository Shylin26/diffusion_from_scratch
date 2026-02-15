import matplotlib.pyplot as plt
from data import get_mnist_loader
from forward import Diffusion

loader=get_mnist_loader()
diffusion=Diffusion()
image, _=next(iter(loader))
plt.figure(figsize=(15,3))
for i,t in enumerate([0,50,100,200,299]):
    xt,_=diffusion.forward(image,t)
    plt.subplot(1,5,i+1)
    plt.imshow(xt.squeeze(),cmap="gray")
    plt.title(f"t={t}")
    plt.axis("off")
plt.show()