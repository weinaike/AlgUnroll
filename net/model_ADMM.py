import torch
import numpy as np
from .model_block import SoftThresh


class ADMM(torch.nn.Module):
    def __init__(self, mode = "only_tv", iter = 200, sp_file = None):
        # 步长

        self.gamma_l1 = 2              # u_l1
        self.gamma_tv = 2              # u_tv
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

        self.iter = iter
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

        elif mode == "l1_tv":
            self.gamma_dwt = 0
            self.beta = 0

            self.lambda_dwt = 0
            self.sigma = 0
        elif mode == "tv_other":
            self.gamma_l1 = 0
            self.gamma_dwt = 0

            self.lambda_l1 = 0
            self.lambda_dwt = 0
        elif mode =="l1_other":
            self.gamma_tv = 0
            self.gamma_dwt = 0

            self.lambda_tv = 0
            self.lambda_dwt = 0
        else:
            print("mode [{}] is not support ".format(mode))
            assert(0)
        if sp_file == None:
            sp_file = "data/SpectralResponse_9.npy"

        self.AT = torch.tensor(np.load(sp_file), dtype=torch.float)
        self.A = torch.transpose(self.AT, 0, 1)

        sz = self.A.size()

        I = torch.eye(sz[1])
        I_roll = torch.roll(I, 1, dims = 1)
        self.Delta = I_roll - I
        self.DeltaT = torch.transpose(self.Delta, 0, 1)
        self.DeltaTDelta = torch.matmul(self.DeltaT, self.Delta)
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
                val = torch.matmul(self.Delta, x) + eta / self.gamma_tv
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
                z = mu_1 * z  + mu_2 * (x + rho/self.beta) # - mu_3 * RegularBlock(z)
            return z
        else:
            return torch.zeros_like(x)

    def update_x(self, b, u_l1, eta_l1, u_tv, eta_tv, u_dwt, eta_dwt, z, rho, w, tau):
        resiual = torch.matmul(self.AT, b) 
        resiual += (torch.mul(self.gamma_l1, u_l1) - eta_l1) 

        z_tv = (torch.mul(self.gamma_tv, u_tv) - eta_tv)
        # z_tv_roll = torch.roll(z_tv, 1, 0)
        # resiual += (z_tv - z_tv_roll)
        resiual += torch.matmul(z_tv, self.DeltaT)
        # resiual += z_tv

        resiual += (torch.mul(self.gamma_dwt, u_dwt) - eta_dwt) 
        resiual += (torch.mul(self.beta, z) - rho) 
        resiual += (torch.mul(self.alpha , w) - tau)
        x =  torch.matmul(self.inv_item, resiual)
        return x
        
    def update_eta(self, x, u, eta, mode = "tv"):
        if mode == "l1":
            eta += torch.mul(self.gamma_l1, (x - u))
        elif mode == "tv":
            eta += torch.mul(self.gamma_tv , (torch.matmul(self.Delta, x) - u))
        elif mode == "dwt":
            eta += torch.mul(self.gamma_dwt , (torch.matmul(self.Delta, x) - u)) ###todo
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
        
        b = torch.matmul(self.A, target)

        x = torch.randn_like(target)
        sz = x.size()

        eta_l1 = torch.zeros_like(x)
        eta_tv = torch.zeros_like(x)
        eta_dwt = torch.zeros_like(x)
        rho = torch.zeros_like(x)
        tau = torch.zeros_like(x)
        for i in range(self.iter):
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


    
    

