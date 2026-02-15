import torch
import torch.nn as nn
class TimeEmbedding(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.embedding=nn.Embedding(300,dim)
    
    def forward(self,t):
        return self.embedding(t)
class NoisePredictor(nn.Module):
    def __init__(self,time_dim=32):
        super().__init__()
        self.time_embed=TimeEmbedding(time_dim)
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(32,1,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
    def forward(self,x,t):
        t_emb=self.time_embed(t) #(batch,time_dim)
        t_emb=t_emb[:,:,None,None] #(batch,time_dim,1,1)
        h=self.relu(self.conv1(x))
        h=h+t_emb[:,:h.shape[1]] #inject time
        out=self.conv3(h)
        return out
    
if __name__ == "__main__":
    model = NoisePredictor()
    x = torch.randn(4, 1, 28, 28)
    t=torch.randint(0,300,(4,))
    y = model(x,t)
    print(y.shape)