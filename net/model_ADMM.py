import torch
import numpy as np
from model_block import SoftThresh


class ADMM(torch.nn.Module):
    def __init__(self, mode = "only_tv"):
        # 步长

        self.gamma_l1 = 10              # u_l1
        self.gamma_tv = 10              # u_tv
        self.gamma_dwt = 1              # u_dwt
        self.alpha = 5                  # w 
        self.beta =  5                  # g(z)        
        self.s = 1.0                    # for g(z)
        ## 拉格朗日乘子
        self.lambda_l1 = 1.0e-4
        self.lambda_tv = 1.0e-4
        self.lambda_dwt = 1.0e-4
        self.sigma = 0.0001             # for g(z)
        self.xi = 0.0001 # I+(w)

       
        if mode == "only_tv":          
            self.gamma_l1 = 0
            self.gamma_dwt = 0
            self.beta = 0            

            self.lambda_l1 = 0
            self.lambda_dwt = 0
            self.sigma = 0
        elif mode == "only_l1":
            self.gamma_tv = 0
            self.gamma_dwt = 0
            self.beta = 0

            self.lambda_tv = 0
            self.lambda_dwt = 0
            self.sigma = 0

        elif mode == "l1+tv":
            self.gamma_dwt = 0
            self.beta = 0

            self.lambda_dwt = 0
            self.sigma = 0
        elif mode == "tv+other":
            self.gamma_l1 = 0
            self.gamma_dwt = 0

            self.lambda_l1 = 0
            self.lambda_dwt = 0
        elif mode =="l1+other":
            self.gamma_tv = 0
            self.gamma_dwt = 0

            self.lambda_tv = 0
            self.lambda_dwt = 0
        else:
            print("mode [{}] is not support ".format(mode))
            assert(0)

        self.A = torch.transpose(torch.tensor(np.load("data/SpectralResponse_9.npy"), dtype=torch.float), 0, 1)
        sz = self.A.size()

        I = torch.eye(sz[1])
        I_roll = torch.roll(I, 1, dims = 1)
        self.Delta = I_roll - I
        self.DeltaTDelta = torch.matmul(torch.transpose(self.Delta, 0, 1), self.Delta)
        self.ATA = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        self.inv_item = torch.linalg.inv(self.ATA + self.gamma_tv * self.DeltaTDelta + (self.alpha + self.beta + self.gamma_l1) * I)

    # mode in ["l1", "tv", "dwt"]
    def update_ui(self, x, eta, mode = "tv"):   
        if mode == "l1":
            if self.lambda_l1 != 0:
                val = x + eta / self.gamma_l1
                thresh = self.lambda_l1 / self.gamma_l1
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "tv":
            if self.lambda_tv != 0:
                x_roll = torch.roll(x, 1, dims = 0)
                delta_x = x_roll - x
                val = delta_x + eta / self.gamma_tv
                thresh = self.lambda_tv / self.gamma_tv
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "dwt":
            if self.lambda_dwt != 0:
                return torch.zeros_like(x) #
            else:
                return torch.zeros_like(x)
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
    
    def update_w(self, x, tau):
        val = x + tau/self.alpha
        return torch.maximum(val,torch.tensor(0))
    
    def update_z(self, x, rho, k_iter = 3):
        if self.sigma != 0:
            mu_1 = (1 - self.beta * self.s)
            mu_2 = self.beta * self.s
            mu_3 = self.sigma * self.s
            z = mu_2 * (x + rho) 
            for i in range(k_iter):
                z = mu_1 * z  + mu_2 * (x + rho) # - mu_3 * RegularBlock(z)
            return z
        else:
            return torch.zeros_like(x)

    def update_x(self, b, u_l1, eta_l1, u_tv, eta_tv, u_dwt, eta_dwt, z, rho, w, tau):
        resiual = torch.matmul(b, self.A) + (torch.mul(self.gamma_l1, u_l1) - eta_l1) \
                  + (torch.mul(self.gamma_tv, u_tv) - eta_tv) + (torch.mul(self.gamma_dwt, u_dwt) - eta_dwt) \
                  + (torch.mul(self.beta, z) - rho) + (torch.mul(self.alpha , w) - tau)
        x =  torch.matmul(resiual, self.inv_item)
        return x
        
    def update_eta(self, x, u, eta, mode = "tv"):
        if mode == "l1":
            eta += torch.mul(self.gamma_l1, (x - u))
        elif mode == "tv":
            eta += torch.mul(self.gamma_tv , (torch.matmul(self.Delta,x) - u))
        elif mode == "dwt":
            eta += torch.mul(self.gamma_dwt , (torch.matmul(self.Delta,x) - u)) ###todo
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
        return eta
    def update_rho(self, x, z, rho):
        rho += torch.mul(self.beta , (x - z))
        return rho

    def update_tau(self, x, w, tau):
        tau += torch.mul(self.alpha, (x - w))
        return tau


    def forward(self, target):
        iter = 50
        b = torch.matmul(self.A, target)

        x = torch.randn_like(target)
        sz = x.size()

        eta_l1 = torch.zeros_like(x)
        eta_tv = torch.zeros_like(x)
        eta_dwt = torch.zeros_like(x)
        rho = torch.zeros_like(x)
        tau = torch.zeros_like(x)
        for i in range(iter):
            u_l1 = self.update_ui(x, eta_l1, mode = "l1")
            u_tv = self.update_ui(x, eta_tv, mode = "tv")
            u_dwt = self.update_ui(x, eta_dwt, mode = "dwt")
            w = self.update_w(x, tau)
            z = self.update_z(x, rho, k_iter=3)
            x = self.update_x(b, u_l1, eta_l1, u_tv, eta_tv, u_dwt, eta_dwt, z, rho, w, tau)

            eta_l1 = self.update_eta(x, u_l1, eta_l1, mode="l1")
            eta_tv = self.update_eta(x, u_tv, eta_tv, mode="tv")
            eta_dwt = self.update_eta(x, u_dwt, eta_dwt, mode="dwt")
            rho = self.update_rho(x, z, rho)
            tau = self.update_tau(x, w, tau)

        return x

if __name__ == '__main__':
    admm = ADMM(mode="l1+tv")
    import random
    center = random.uniform(400,3500)
    # center = random.uniform(1000,1100)
    sigma = random.uniform(50,100)
    sp = np.load("data/SpectralResponse_9.npy")
    length, det_num = sp.shape
    x = np.linspace(355,3735,length)
    spectral = np.exp(-1 * (x - center)**2 / (sigma**2 ))

    rex = admm.forward(torch.tensor(spectral,dtype=torch.float))
    import matplotlib.pylab as plt
    # plt.rcParams['figure.figsize'] = (8.0, 4.0) 
    plt.figure()
    plt.plot(x, spectral,'b', x, rex,'r:')
    plt.show()
