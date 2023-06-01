
import torch
import numpy as np
from .model_block import SoftThresh, RegularBlock


class LADMM(torch.nn.Module):
    def __init__(self, mode = "only_tv", sp_file = "data/SpectralResponse_9.npy"):
        super(LADMM, self).__init__()
        # 步长
        self.alpha = torch.nn.Parameter(torch.Tensor([5]))                   # w 
        self.beta =  torch.nn.Parameter(torch.Tensor([5]))                   # g(z)
        self.gamma_l1 = torch.nn.Parameter(torch.Tensor([10]))               # u_l1
        self.gamma_tv = torch.nn.Parameter(torch.Tensor([10]))               # u_tv
        self.gamma_dwt = torch.nn.Parameter(torch.Tensor([1]))               # u_dwt
        self.s = torch.nn.Parameter(torch.Tensor([1.0]))                     # for g(z)
        ## 拉格朗日乘子
        self.lambda_l1 = torch.nn.Parameter(torch.Tensor([1.0e-4])) 
        self.lambda_tv = torch.nn.Parameter(torch.Tensor([1.0e-4])) 
        self.lambda_dwt = torch.nn.Parameter(torch.Tensor([1.0e-4])) 
        self.sigma = torch.nn.Parameter(torch.Tensor([1.0e-4]))              # for g(z)
        self.xi = torch.nn.Parameter(torch.Tensor([1.0e-4]))                 # I+(w)

        self.enble_l1 = 1
        self.enble_tv = 1
        self.enble_dwt = 1
        self.enble_cnn = 1
       
        if mode == "only_tv":          
            self.enble_l1 = 0
            self.enble_tv = 1
            self.enble_dwt = 0
            self.enble_cnn = 0
        elif mode == "only_l1":
            self.enble_l1 = 1
            self.enble_tv = 0
            self.enble_dwt = 0
            self.enble_cnn = 0

        elif mode == "l1_tv":
            self.enble_l1 = 1
            self.enble_tv = 1
            self.enble_dwt = 0
            self.enble_cnn = 0
        elif mode == "tv_cnn":
            self.enble_l1 = 0
            self.enble_tv = 1
            self.enble_dwt = 0
            self.enble_cnn = 1
        elif mode =="l1_cnn":
            self.enble_l1 = 1
            self.enble_tv = 0
            self.enble_dwt = 0
            self.enble_cnn = 1
        else:
            print("mode [{}] is not support ".format(mode))
            assert(0)

        self.A = torch.transpose(torch.tensor(np.load(sp_file), dtype=torch.float), 0, 1)
        sz = self.A.size()

        self.I = torch.eye(sz[1])
        I_roll = torch.roll(self.I, 1, dims = 1)
        self.Delta = I_roll - self.I
        self.DeltaTDelta = torch.matmul(torch.transpose(self.Delta, 0, 1), self.Delta)
        self.ATA = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        

    # mode in ["l1", "tv", "dwt"]
    def update_ui(self, x, eta, mode = "tv"):   
        if mode == "l1":
            if self.enble_l1 != 0:
                val = x + eta / self.gamma_l1
                thresh = self.lambda_l1 / self.gamma_l1
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "tv":
            if self.enble_tv != 0:
                x_roll = torch.roll(x, 1, dims = 1)
                delta_x = x_roll - x
                val = delta_x + eta / self.gamma_tv
                thresh = self.lambda_tv / self.gamma_tv
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "dwt":
            if self.enble_dwt != 0:
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
        if self.enble_cnn == 1:
            mu_1 = (1 - self.beta * self.s)
            mu_2 = self.beta * self.s
            mu_3 = self.sigma * self.s
            z = mu_2 * (x + rho) 
            for i in range(k_iter):
                z = mu_1 * z  + mu_2 * (x + rho) - mu_3 * RegularBlock(z)
            return z
        else:
            return torch.zeros_like(x)

    def update_x(self, b, u_l1, eta_l1, u_tv, eta_tv, u_dwt, eta_dwt, z, rho, w, tau):
        
        resiual = torch.matmul(b, self.A) + (torch.mul(self.alpha , w) - tau)
        add_item = self.ATA + self.alpha * self.I
        if self.enble_l1 == 1:
            resiual += (torch.mul(self.gamma_l1, u_l1) - eta_l1) 
            add_item += self.gamma_l1 * self.I
        if self.enble_tv == 1:
            resiual += (torch.mul(self.gamma_tv, u_tv) - eta_tv) 
            add_item += self.gamma_tv * self.DeltaTDelta
        if self.enble_dwt == 1:
            resiual += (torch.mul(self.gamma_dwt, u_dwt) - eta_dwt) 
            add_item += self.gamma_dwt * self.DeltaTDelta               #todo
        if self.enble_cnn == 1:
            resiual += (torch.mul(self.beta, z) - rho) 
            add_item += self.beta * self.I
        
        inv_item = torch.linalg.inv(add_item)
        x =  torch.matmul(resiual, inv_item)
        return x
        
    def update_eta(self, x, u, eta, mode = "tv"):
        if mode == "l1":
            if self.enble_l1 == 1:
                eta += torch.mul(self.gamma_l1, (x - u))
        elif mode == "tv":
            if self.enble_tv == 1:
                eta += torch.mul(self.gamma_tv , (torch.matmul(self.Delta,x) - u))
        elif mode == "dwt":
            if self.enble_dwt == 1:
                eta += torch.mul(self.gamma_dwt , (torch.matmul(self.Delta,x) - u)) ###todo
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
        return eta
    
    def update_rho(self, x, z, rho):
        if self.enble_cnn == 1:
            rho += torch.mul(self.beta , (x - z))
        return rho

    def update_tau(self, x, w, tau):
        tau += torch.mul(self.alpha, (x - w))
        return tau


    def forward(self, b, target):
        iter = 5

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