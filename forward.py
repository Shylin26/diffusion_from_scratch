import torch
class Diffusion:
    def __init__(self,timesteps=300,beta_start=1e-4,beta_end=0.02,device="cpu"):
        self.device=device
        self.T=timesteps
        self.betas=torch.linspace(beta_start,beta_end,timesteps,device=device)
        self.alphas=1.0-self.betas
        self.alpha_bars=torch.cumprod(self.alphas,dim=0)

    def forward(self,x0,t):
        noise=torch.randn_like(x0)
        alpha_bar_t=self.alpha_bars[t]
        #reshape from broadcasting
        alpha_bar_t=alpha_bar_t[:,None,None,None]
        xt=(
            torch.sqrt(alpha_bar_t)*x0+
            torch.sqrt(1-alpha_bar_t)*noise
        )
        return xt,noise

    def sample_step(self,model,xt,t):
        beta_t=self.betas[t]
        alpha_t=self.alphas[t]
        alpha_bar_t=self.alpha_bars[t]

        beta_t=beta_t[:,None,None,None]
        alpha_t=alpha_t[:,None,None,None]
        alpha_bar_t=alpha_bar_t[:,None,None,None]

        eps_pred=model(xt,t)
        # mean of p(x_{t-1}|x_t)
        mean=(1/torch.sqrt(alpha_t)) *(
            xt-(beta_t/torch.sqrt(1-alpha_bar_t))*eps_pred
        )
        if t[0]==0:
            return mean
        noise=torch.randn_like(xt)
        return mean+torch.sqrt(beta_t)*noise