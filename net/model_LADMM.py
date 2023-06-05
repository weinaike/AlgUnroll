
import torch
import numpy as np
from .model_block import SoftThresh, RegularBlock


class LADMM(torch.nn.Module):
    def __init__(self, mode = "only_tv", sp_file = "data/SpectralResponse_9.npy", iter = 5, filter = 32, ks = 3):
        super(LADMM, self).__init__()
        self.layer_num = iter
        # 步长(二次惩罚项惩罚因子)
        val_init = 5.0
        self.gamma_l1 = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*2 ))               # u_l1
        self.gamma_tv = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*10 ))               # u_tv
        self.gamma_dwt = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ))               # u_dwt
        self.alpha = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*10 ))                   # w 
        self.beta =  torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*1 ))                   # g(z)

        self.s = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*0.5))                     # for g(z)


        ## 拉格朗日乘子
        val_init = 1.0e-3
        self.lambda_l1 = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init )) 
        self.lambda_tv = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init )) 
        self.lambda_dwt = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init )) 
        self.sigma = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ))              # for g(z)
        self.xi = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ))                 # I+(w)

        self.A = torch.transpose(torch.tensor(np.load(sp_file), dtype=torch.float), 0, 1)
        sz = self.A.size()

        self.I = torch.eye(sz[1])
        I_roll = torch.roll(self.I, 1, dims = 1)
        self.Delta = I_roll - self.I
        self.DeltaT = torch.transpose(self.Delta, 0, 1)
        self.DeltaTDelta = torch.matmul(self.DeltaT, self.Delta)
        self.ATA = torch.matmul(torch.transpose(self.A, 0, 1), self.A)
        
        self.enble_l1 = 1
        self.enble_tv = 1
        self.enble_dwt = 1
        self.enble_cnn = 1
        self.set_enble(mode)

        self.k_iter = 3
        self.blocks = list()
        for i in range(iter * self.k_iter):
            self.blocks.append(RegularBlock(filter, ks))

    # set regular enble
    def set_enble(self,mode):
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
        elif mode =="l1_tv_cnn":
            self.enble_l1 = 1
            self.enble_tv = 1
            self.enble_dwt = 0
            self.enble_cnn = 1
        elif mode =="only_cnn":
            self.enble_l1 = 0
            self.enble_tv = 0
            self.enble_dwt = 0
            self.enble_cnn = 1
        else:
            print("mode [{}] is not support ".format(mode))
            assert(0)

    # mode in ["l1", "tv", "dwt"]
    def update_ui(self, i, x, eta, mode = "tv"):   
        if mode == "l1":
            if self.enble_l1 == 1:
                val = x + eta / torch.abs(self.gamma_l1[i])
                thresh = torch.abs(self.lambda_l1[i] / self.gamma_l1[i])
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "tv":
            if self.enble_tv == 1:
                val = torch.matmul(x, self.Delta) + eta / torch.abs(self.gamma_tv[i])
                thresh = torch.abs(self.lambda_tv[i] / self.gamma_tv[i])
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "dwt":
            if self.enble_dwt == 1:
                return torch.zeros_like(x) #
            else:
                return torch.zeros_like(x)
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
    
    def update_w(self, i, x, tau):
        val = x + tau/torch.abs(self.alpha[i])
        return torch.maximum(val,torch.tensor(0))
    
    def update_z(self, i,  x, rho, k_iter = 3):
        if self.enble_cnn == 1:
            # mu_1 = (1 - self.beta[i] * self.s[i])
            # mu_2 = (self.beta[i] * self.s[i])
            # mu_3 = (self.simga[i] * self.s[i])
            mu_2 = torch.abs(self.s[i])
            mu_3 = torch.abs(self.sigma[i])
            z = x + rho / torch.abs(self.beta[i])
            for j in range(k_iter):
                z = (1 - mu_2) * z  + mu_2 * (x + rho / self.beta[i]) - mu_3 * self.blocks[i * k_iter + j](z)
                # print(z)
            return z
        else:
            return torch.zeros_like(x)

    def update_x(self, i, b, u_l1, eta_l1, u_tv, eta_tv, u_dwt, eta_dwt, z, rho, w, tau):
        
        resiual = torch.matmul(b, self.A) + torch.abs(self.alpha[i]) * w - tau
        add_item = self.ATA + torch.abs(self.alpha[i]) * self.I
        if self.enble_l1 == 1:
            resiual += torch.abs(self.gamma_l1[i]) * u_l1 - eta_l1
            add_item += torch.abs(self.gamma_l1[i]) * self.I
        if self.enble_tv == 1:
            resiual += torch.matmul(torch.abs(self.gamma_tv[i]) * u_tv - eta_tv, self.DeltaT)
            add_item += torch.abs(self.gamma_tv[i]) * self.DeltaTDelta
        if self.enble_dwt == 1:
            resiual += torch.abs(self.gamma_dwt[i]) * u_dwt - eta_dwt
            add_item += torch.abs(self.gamma_dwt[i]) * self.DeltaTDelta               #todo
        if self.enble_cnn == 1:
            resiual += torch.abs(self.beta[i]) * z - rho
            add_item += torch.abs(self.beta[i]) * self.I
        
        inv_item = torch.linalg.inv(add_item)
        x =  torch.matmul(resiual, inv_item)
        return x
        
    def update_eta(self, i, x, u, eta, mode = "tv"):
        eta_n = eta
        if mode == "l1":
            if self.enble_l1 == 1:
                eta_n = eta + torch.abs(self.gamma_l1[i]) * (x - u)
        elif mode == "tv":
            if self.enble_tv == 1:
                eta_n = eta + torch.abs(self.gamma_tv[i]) * (torch.matmul(x, self.Delta) - u)
        elif mode == "dwt":
            if self.enble_dwt == 1:
                eta_n = eta + torch.abs(self.gamma_dwt[i]) * (torch.matmul(x, self.Delta) - u) ###todo
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
        return eta_n
    
    def update_rho(self, i,  x, z, rho):
        rho_n = rho
        if self.enble_cnn == 1:
            rho_n = rho + torch.abs(self.beta[i]) * (x - z)
        return rho_n

    def update_tau(self, i, x, w, tau):
        tau_n = tau + torch.abs(self.alpha[i]) * (x - w)
        return tau_n


    def forward(self, b, target):
        x =  torch.ones_like(target)

        eta_l1 = torch.zeros_like(x)
        eta_tv = torch.zeros_like(x)
        eta_dwt = torch.zeros_like(x)
        rho = torch.zeros_like(x)
        tau = torch.zeros_like(x)

        for i in range(self.layer_num):
            u_l1 = self.update_ui(i, x, eta_l1, mode = "l1")
            u_tv = self.update_ui(i, x, eta_tv, mode = "tv")
            u_dwt = self.update_ui(i, x, eta_dwt, mode = "dwt")
            w = self.update_w(i, x, tau)
            z = self.update_z(i, x, rho, k_iter=self.k_iter)
            x = self.update_x(i, b, u_l1, eta_l1, u_tv, eta_tv, u_dwt, eta_dwt, z, rho, w, tau)
            

            eta_l1 = self.update_eta(i, x, u_l1, eta_l1, mode="l1")
            eta_tv = self.update_eta(i, x, u_tv, eta_tv, mode="tv")
            eta_dwt = self.update_eta(i, x, u_dwt, eta_dwt, mode="dwt")
            rho = self.update_rho(i, x, z, rho)
            tau = self.update_tau(i, x, w, tau)
        
        return x
    def to(self, device):
        super(LADMM, self).to(device)
        self.A = self.A.to(device)
        self.I = self.I.to(device)
        self.Delta = self.Delta.to(device)
        self.DeltaT = self.DeltaT.to(device)
        self.DeltaTDelta = self.DeltaTDelta.to(device)
        self.ATA = self.ATA.to(device)

        for block in self.blocks:
            block.to(device)

        return self