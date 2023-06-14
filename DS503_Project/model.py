from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from ddpm import DDPM

class Model() :
    """description of class"""
    def __init__(self, device = None, T = 1000, beta = None, n_epochs : int = 20, image_shape = (1, 28, 28) ):
        self.device_ = device
        self.T_ = T 
        self.n_epochs_ = n_epochs
        self.image_shape_ = image_shape

        if beta is None or beta == 'linear' :
            self.betas_ = torch.linspace(start = 1e-5, end = 1e-2,  steps = self.T_)
        elif beta == 'interpolate_linear' :
            self.betas_ = torch.linspace(start = 1e-5, end = 1e-2,  steps = 1000)
        

        self.betas_ = self.betas_.to(self.device_)
        self.alphas_ = 1 - self.betas_.to(self.device_)
        self.alphas_prod_ = torch.cumprod(self.alphas_, 0).to(self.device_)
        self.alphas_bar_sqrt_ = torch.sqrt(self.alphas_prod_).to(self.device_)
        self.one_minus_alphas_bar_sqrt_ = torch.sqrt(1 - self.alphas_prod_).to(self.device_)

        self.model_ = DDPM(device = self.device_, betas = self.betas_, T = self.T_, image_shape= self.image_shape_ )
    
    def set_x(self, x) :
        self.x_ = x

    def q_sample(self, x_0) :
        return self.model_(x_0, torch.tensor([self.T_ - 1]).repeat(x_0.shape[0]).to(self.device_)  )

    def fit(self, save : bool = False, file_name : str = 'ddpm') :
        mse = nn.MSELoss()
        best_loss = float("inf")
        optimizer = optim.Adam(self.model_.parameters())

        for epoch in tqdm(range(self.n_epochs_), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            for step, batch in enumerate(tqdm(self.x_, leave=False, desc=f"Epoch {epoch + 1}/{self.n_epochs_}", colour="#005500")):
                # Loading data
                
                x0 = batch.to(self.device_)
                
                n = len(x0)

                # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
                eta = torch.randn_like(x0).to(self.device_)
                t = torch.randint(0, self.T_, (n,)).to(self.device_)

                # Computing the noisy image based on x0 and the time-step (forward process)
                noisy_imgs = self.model_(x0, t, eta)

                # Getting model estimation of noise based on the images and the time-step
                eta_theta = self.model_.backward(noisy_imgs, t.reshape(n, -1))

                # Optimizing the MSE between the noise plugged and the predicted noise
                loss = mse(eta_theta, eta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(x0) / len(self.x_.dataset)

            log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

            # Storing the model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                if save : torch.save(self.model_.state_dict(), f = 'models/'+ file_name + '_epoch' + str(epoch) + '.pt' )
                log_string += " --> Best model ever (stored)"
            print(log_string)

    def p_sample(self, x_T = None, n_samples=4):
        """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
        c, h, w = self.image_shape_
        with torch.no_grad():
            #if device is None:
            #    device = self.model_.device
            
            # Starting from random noise
            if x_T is None :
                x = torch.randn(n_samples, c, h, w ).to(self.device_)
            else :
                x = x_T

            for idx, t in tqdm(enumerate(list(range(self.T_))[::-1])):
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(self.device_).long()
                eta_theta = self.model_.backward(x, time_tensor)

                alpha_t = self.model_.alphas[t]
                alpha_t_bar = self.model_.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, c, h, w).to(self.device_)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = self.model_.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Option 2: sigma_t squared = beta_tilda_t
                    # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    # sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

        return x

    def model_load(self, file_name : str) : 
        self.model_ = DDPM(device = self.device_, betas = self.betas_, image_shape = self.image_shape_)
        self.model_.load_state_dict(torch.load('models/' + file_name + '.pt', map_location=self.device_))
        self.model_.eval()
        print('Model loaded')

    def multi_interpolate_sample(self, x, y, image_size = (1, 28, 28) ) :
        dat = self.q_sample(x)
        dat_2 = self.q_sample(y)
        k = torch.empty( (1, *image_size)).to(self.device_)
        for i in dat : 
            temp = torch.cat( (i.reshape(-1, *image_size), dat_2) )
            temp = (temp / temp.shape[0]).sum(axis = 0 , keepdim = True)
            #temp = temp.sum(axis = 0, keepdim = True)
            k = torch.cat( (k, temp  ) ,dim = 0 )
        k = k[1:, :, :, :]

        return self.p_sample(k, n_samples = dat.shape[0])