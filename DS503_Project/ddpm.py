import torch
import torch.nn as nn
from u_net import UNet

def extract(input, t, x):
    """
    This is just a utility function that helps DDPM class. you don't need to change anything.
    """
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class DDPM(nn.Module):
    """
    This is DDPM class. You don't need to change anything here.
    """
    def __init__(self, betas, network = None, T = 1000, device = None, image_shape=(1, 28, 28) ) :
        """
        betas --> This is (n * 1) vector that contains beta_t. You can create betas with BetaGenerator class.
        network --> U-Net is default.
        T --> How many steps are in diffusion process? 'T = 1000' is dafault.
        device --> CUDA or CPU.
        image_shape --> It's a sequence that has three components. FIrst is channel of the image, second is height, third is width. 
                        For example, the image shape will be (1, 28, 28) if your data is MNIST data and it's default.
        """
        super(DDPM, self).__init__()
        self.T_ = T
        self.device_ = device
        self.image_shape_ = image_shape
        if network is None :
            self.network_ = UNet(h = image_shape[1] , w = image_shape[2]).to(device)
        else :
            self.network_ = network.to(device)

        self.betas = betas
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)

        # Get number of images, channel, height, width
        if len(x0.shape) == 4 :
            n, c, h, w = x0.shape
        else :
            n, h, w = x0.shape
            c = 1

        a_bar = extract(self.alpha_bars, t, x0)

        # Random noise
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device_)
        
        # Noise-added image
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network_(x, t)