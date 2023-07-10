
import torch
import numpy as np
import PIL.Image as Image
from .model_block import SoftThresh, RegularBlock
import math
def closest_power_of_two(n):
    power = math.floor(math.log2(n)) + 1
    result = 2 ** power
    return result

class LADMM(torch.nn.Module):
    def __init__(self, mode = "only_tv", psf_file = "data/pdf.tiff", iter = 5,  senor_size =[480,270], filter = 32, ks = 3):
        super(LADMM, self).__init__()
        self.layer_num = iter
        # 步长(二次惩罚项惩罚因子)
        val_init = 5.0e-5
        grad = True
        self.alpha = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)                  # w 
        self.beta =  torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)                  # g(z)        
        self.gamma_l1 = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)               # u_l1
        self.gamma_tv = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)               # u_tv
        self.delta =  torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)                 # Mx
        self.s = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init), requires_grad=grad)                       # for g(z)

        ## 拉格朗日乘子
        val_init = 5.0e-5
        self.lambda_l1 = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad) 
        self.lambda_tv = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad) 
        self.sigma = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)              # for g(z)
        self.xi = torch.nn.Parameter(torch.Tensor(torch.ones(self.layer_num)*val_init ), requires_grad=grad)                 # I+(w)
        
        
        im = Image.open(psf_file)
        im = im.resize(senor_size)
        psf = np.array(im,dtype='float32')
        h, w, c = psf.shape
        for i in range(c):        
            psf[:,:,i] /= np.linalg.norm(psf[:,:,i].ravel())
        psf = torch.tensor(psf).permute(2,0,1)  
        

        self.c = c
        self.sz = torch.Size([h, w])
        #self.full_sz = torch.Size([closest_power_of_two(self.sz[0]*2), closest_power_of_two(self.sz[1]*2)])
        self.full_sz = torch.Size([self.sz[0]*2, self.sz[1]*2])

        self.H_fft = torch.fft.fft2(torch.fft.ifftshift(self.Pad(psf), dim=(-2, -1)))
        self.MTM = (torch.abs(torch.conj(self.H_fft)*self.H_fft)) #这个需要再探讨
        self.DeltaTDelta = self.precompute_DeltaTDelta()
        
        self.enble_l1 = 1
        self.enble_tv = 1
        self.enble_cnn = 1
        self.set_enble(mode)

        self.param = self.set_param_list()
        self.k_iter = 3
        if self.enble_cnn == 1:            
            self.blocks = torch.nn.ModuleList()
            for i in range(iter * self.k_iter):
                self.blocks.append(RegularBlock(filter, ks))
        

        #self.has_res = has_res
        #if has_res:
        #    pass
            #self.res = CausalSelfAttention(num_heads=4, embed_dimension=sz[1], bias=False, is_causal=True, dropout=0.1)

    def set_param_list(self):
        param = torch.nn.ParameterList()
        param.append(self.gamma_l1)
        param.append(self.gamma_tv)
        param.append(self.alpha)
        param.append(self.beta)
        param.append(self.delta)
        param.append(self.s)
        param.append(self.lambda_l1)
        param.append(self.lambda_tv)
        param.append(self.sigma)
        param.append(self.xi)
        return param


    # set regular enble
    def set_enble(self,mode):
        if mode == "only_tv":          
            self.enble_l1 = 0
            self.enble_tv = 1
            self.enble_cnn = 0
        elif mode == "only_l1":
            self.enble_l1 = 1
            self.enble_tv = 0
            self.enble_cnn = 0
        elif mode == "l1_tv":
            self.enble_l1 = 1
            self.enble_tv = 1
            self.enble_cnn = 0
        elif mode == "tv_cnn":
            self.enble_l1 = 0
            self.enble_tv = 1
            self.enble_cnn = 1
        elif mode =="l1_tv_cnn":
            self.enble_l1 = 1
            self.enble_tv = 1
            self.enble_cnn = 1
        elif mode =="only_cnn":
            self.enble_l1 = 0
            self.enble_tv = 0
            self.enble_cnn = 1
        else:
            print("mode [{}] is not support ".format(mode))
            assert(0)

    def precompute_DeltaTDelta(self):
        PsiTPsi = torch.zeros(self.full_sz)
        PsiTPsi[0,0] = 4
        PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
        PsiTPsi = np.abs(torch.fft.fft2(PsiTPsi))
        return PsiTPsi.repeat(self.c,1,1)
    
    def Delta(self, img):
        return torch.stack((torch.roll(img,1,dims=2) - img, torch.roll(img,1,dims=3) - img), dim=4)

    def DeltaT(self, x_diff):
        vec1 = x_diff[:,:,:,:,0]
        vec2 = x_diff[:,:,:,:,1]
        diff1 = torch.roll(vec1, -1, dims=2) - vec1
        diff2 = torch.roll(vec2, -1, dims=3) - vec2
        return diff1 + diff2    
    
    def Crop(self,M):
        # Image stored as matrix (row-column rather than x-y)
        top = (self.full_sz[0] - self.sz[0])//2
        bottom = (self.full_sz[0] + self.sz[0])//2
        left = (self.full_sz[1] - self.sz[1])//2
        right = (self.full_sz[1] + self.sz[1])//2
        return M[:,:,top:bottom,left:right]

    def Pad(self,b):
        v_pad = (self.full_sz[0] -  self.sz[0])//2
        h_pad = (self.full_sz[1] -  self.sz[1])//2
        return torch.nn.functional.pad(b,(h_pad,h_pad,v_pad,v_pad),"constant",0)
    
    def PSF(self, x):
        H_fft = self.H_fft.repeat(x.size()[0],1,1,1)
        Mx = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.ifftshift(x,dim=(-2,-1))) * H_fft), dim=(-2,-1)))
        return Mx
    
    def conjPSF(self,x):        
        H_fft = self.H_fft.repeat(x.size()[0],1,1,1)
        MTx = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2,-1))) * torch.conj(H_fft)),dim=(-2,-1)))
        return MTx

    # mode in ["l1", "tv", "dwt"]
    def update_ui(self, i, x, eta, mode = "tv"):   
        if mode == "l1":
            if self.enble_l1 == 1:
                val = x + eta / self.gamma_l1[i]
                thresh =  self.lambda_l1[i] / self.gamma_l1[i]
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(x)
        elif mode == "tv":
            if self.enble_tv == 1:
                val = self.Delta(x) + eta / self.gamma_tv[i]
                thresh =  self.lambda_tv[i] / self.gamma_tv[i]
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(self.Delta(x))
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
    
    def update_w(self, i, x, tau):
        val = x + tau/ self.alpha[i]
        return torch.maximum(val,torch.tensor(0))
    
    def update_z(self, i,  x, rho, k_iter = 3):
        if self.enble_cnn == 1:
            # mu_1 = (1 - self.beta[i] * self.s[i])
            # mu_2 = (self.beta[i] * self.s[i])
            # mu_3 = (self.simga[i] * self.s[i])
            mu_2 =  (self.s[i])
            mu_3 =  (self.sigma[i])
            z = x + rho /  (self.beta[i])
            for j in range(k_iter):
                z = (1 - mu_2) * z  + mu_2 * (x + rho / self.beta[i])# - mu_3 * self.blocks[i * k_iter + j](z)
                # print(z)
            return z
        else:
            return torch.zeros_like(x)

    def update_x(self, i, b, u_l1, eta_l1, u_tv, eta_tv, z, rho, w, tau, v, theta):
        resiual = self.conjPSF(self.delta[i] * v - theta)
        resiual +=  self.alpha[i] * w - tau
        
        batch = b.size()[0]
        MTM = self.MTM.repeat(batch,1,1,1)
        DTD = self.DeltaTDelta.repeat(batch,1,1,1)
        add_item = self.delta[i] * MTM + self.alpha[i]
        if self.enble_l1 == 1:
            resiual +=  self.gamma_l1[i] * u_l1 - eta_l1
            add_item += self.gamma_l1[i]
        if self.enble_tv == 1:
            resiual += self.DeltaT(self.gamma_tv[i] * u_tv - eta_tv)
            add_item +=  self.gamma_tv[i] * DTD
        if self.enble_cnn == 1:
            resiual +=  self.beta[i] * z - rho
            add_item += self.beta[i]
        freq_space_result = torch.fft.fft2(torch.fft.ifftshift(resiual,  dim=(-2,-1)))
        x =  torch.real(torch.fft.fftshift(torch.fft.ifft2( 1.0/add_item *freq_space_result), dim=(-2,-1)))
        return x
    

    def update_v(self, i, x, b, theta):
        v = b + theta + self.delta[i] * self.PSF(x)

        inv = 1/(self.Pad(self.Crop(torch.ones_like(v))) + self.delta[i])
        v = torch.mul(inv, v)
        return v
    
            
    def update_eta(self, i, x, u, eta, mode = "tv"):
        eta_n = eta
        if mode == "l1":
            if self.enble_l1 == 1:
                eta_n = eta +  (self.gamma_l1[i]) * (x - u)
        elif mode == "tv":
            if self.enble_tv == 1:
                eta_n = eta + torch.mul(self.gamma_tv[i] , self.Delta(x) - u)
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
        return eta_n
    
    def update_rho(self, i, x, z, rho):
        rho_n = rho
        if self.enble_cnn == 1:
            rho_n = rho +  (self.beta[i]) * (x - z)
        return rho_n

    def update_tau(self, i, x, w, tau):
        tau_n = tau +  (self.alpha[i]) * (x - w)
        return tau_n
    
    def update_theta(self, i, x, v, theta):
        theta += self.delta[i] * (self.PSF(x) - v)
        return theta

    def forward(self, b):
        b = self.Pad(b)
        x = torch.zeros_like(b)
        eta_l1 = torch.zeros_like(x)
        eta_tv = torch.zeros_like(self.Delta(x))
        rho = torch.zeros_like(x)
        tau = torch.zeros_like(x)
        theta = torch.zeros_like(x)


        u_l1 = torch.zeros_like(x)
        u_tv = torch.zeros_like(self.Delta(x))
        v = torch.zeros_like(x)
        w = torch.zeros_like(x)
        z = torch.zeros_like(x)


        for i in range(self.layer_num):
            u_l1 = self.update_ui(i, x, eta_l1, mode = "l1")
            u_tv = self.update_ui(i, x, eta_tv, mode = "tv")
            
            v = self.update_v(i, x, b, theta)   
            w = self.update_w(i, x, tau)
            z = self.update_z(i, x, rho, k_iter=self.k_iter)
            x = self.update_x(i, b, u_l1, eta_l1, u_tv, eta_tv, z, rho, w, tau, v, theta)
            

            eta_l1 = self.update_eta(i, x, u_l1, eta_l1, mode="l1")
            eta_tv = self.update_eta(i, x, u_tv, eta_tv, mode="tv")
            theta  = self.update_theta(i, x, v, theta)
            rho = self.update_rho(i, x, z, rho)
            tau = self.update_tau(i, x, w, tau)
        
        return  self.Crop(x)
    

    
    def to(self,device):
        super(LADMM, self).to(device)
        self.H_fft = self.H_fft.to(device)
        self.MTM = self.MTM.to(device) 
        self.DeltaTDelta = self.DeltaTDelta.to(device)


