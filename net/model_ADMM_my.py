#整体而言， ||Ax-b||不拆开的效果不然拆开的效果
import torch
import numpy as np
from model_block import SoftThresh
import PIL.Image as Image
import matplotlib.pyplot as plt
class ADMM(torch.nn.Module):
    def __init__(self, psf_file, mode = "tv", iters = 100, senor_size =[480,270], rgb_idx = 0, disp = 20, autotune = False ):
        # 步长
        self.iter = iters
        self.disp = disp
        self.gamma_l1 = 0.01                # u_l1
        self.gamma_tv = 0.02               # u_tv
        self.alpha = 0.1                # w 
  
        ## 拉格朗日乘子
        self.lambda_l1 = 1             # u_l1
        self.lambda_tv = 0.01            # u_tv

        if mode == "tv":          
            self.gamma_l1 = 0
            self.lambda_l1 = 0

        elif mode == "l1":
            self.gamma_tv = 0
            self.lambda_tv = 0

        elif mode == "l1_tv":
            pass
        else:
            print("mode [{}] is not support ".format(mode))
            assert(0)
        im = Image.open(psf_file)
        im = im.resize(senor_size)
        psf = np.array(im,dtype='float32')
        psf = psf[:,:,rgb_idx]
        psf /= np.linalg.norm(psf.ravel(),ord=2) # 二范数归一化，均值在0.001左右        
        psf = torch.tensor(psf)        
        self.sz = psf.size()
        self.full_sz = torch.Size([self.sz[0]*2, self.sz[1]*2])

        self.H_fft = torch.fft.fft2(torch.fft.ifftshift(self.Pad(psf)))

        self.ATA = torch.abs(torch.conj(self.H_fft)*self.H_fft)


        self.DeltaTDelta = self.precompute_DeltaTDelta()

        self.inv_item = 1.0/(self.ATA + self.gamma_tv * self.DeltaTDelta + (self.alpha + self.gamma_l1))

        self.autotune = autotune

        self.res_tol = 1.5
        self.mu_inc  = 1.2
        self.mu_dec  = 1.2


    def update_inv(self):
        self.inv_item = 1.0/(self.ATA + self.gamma_tv * self.DeltaTDelta + (self.alpha + self.gamma_l1))

    def update_param(self, mu, primal_res, dual_res):
        if primal_res > self.res_tol * dual_res:
            mu_up = mu*self.mu_inc
        else:
            mu_up = mu
            
        if dual_res > self.res_tol*primal_res:
            mu_up = mu_up/self.mu_dec
        else:
            mu_up = mu_up
        return mu_up
   

    def precompute_DeltaTDelta(self):
        PsiTPsi = torch.zeros(self.full_sz)
        PsiTPsi[0,0] = 4
        PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
        PsiTPsi = np.abs(torch.fft.fft2(PsiTPsi))
        return PsiTPsi
    
    def Delta(self, img):
        return torch.stack((torch.roll(img,1,dims=0) - img, torch.roll(img,1,dims=1) - img), dim=2)              

    def DeltaT(self, x_diff):
        vec1 = x_diff[:,:,0]
        vec2 = x_diff[:,:,1]
        diff1 = torch.roll(vec1, -1, dims=0) - vec1
        diff2 = torch.roll(vec2, -1, dims=1) - vec2
        return diff1 + diff2    
    
    def Crop(self,M):
        # Image stored as matrix (row-column rather than x-y)
        top = (self.full_sz[0] - self.sz[0])//2
        bottom = (self.full_sz[0] + self.sz[0])//2
        left = (self.full_sz[1] - self.sz[1])//2
        right = (self.full_sz[1] + self.sz[1])//2
        return M[top:bottom,left:right]

    def Pad(self,b):
        v_pad = (self.full_sz[0] -  self.sz[0])//2
        h_pad = (self.full_sz[1] -  self.sz[1])//2
        return torch.nn.functional.pad(b,(h_pad,h_pad,v_pad,v_pad),"constant",0)
    
    def PSF(self, x):
        Ax = torch.real(torch.fft.fftshift(torch.fft.ifft2((torch.fft.fft2(torch.fft.ifftshift(x)) * self.H_fft))))
        return Ax


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
                val = self.Delta(x) + eta / self.gamma_tv
                thresh = self.lambda_tv / self.gamma_tv
                return SoftThresh(val, thresh)
            else:
                return torch.zeros_like(self.Delta(x))
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
    
    def update_w(self, x, tau):
        val = x + tau/self.alpha
        return torch.maximum(val,torch.tensor(0))
    

    def update_x(self, b, u_l1, eta_l1, u_tv, eta_tv, w, tau):
        resiual = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.ifftshift(b)) * torch.conj(self.H_fft))))
        # resiual = self.Pad(self.Crop(resiual))
        if self.gamma_l1 !=0 :
            resiual += (torch.mul(self.gamma_l1, u_l1) - eta_l1) 
        if self.gamma_tv != 0:
            z_tv = (torch.mul(self.gamma_tv, u_tv) - eta_tv)
            resiual += self.DeltaT(z_tv)
        if self.alpha != 0:
            resiual += (torch.mul(self.alpha , w) - tau)

        freq_space_result = self.inv_item * torch.fft.fft2(torch.fft.ifftshift(resiual))

        x =  torch.real(torch.fft.fftshift(torch.fft.ifft2(freq_space_result)))
        return x
        

    def update_eta(self, x, u, eta, mode = "tv"):
        if mode == "l1":
            eta += torch.mul(self.gamma_l1, (x - u))
        elif mode == "tv":
            eta += torch.mul(self.gamma_tv , self.Delta(x) - u)
        else:
            print("mode [{}] is not support".format(mode))
            assert(0)
        return eta


    def update_tau(self, x, w, tau):
        tau += torch.mul(self.alpha, (x - w))
        return tau
   
    def forward(self, b):

        b = self.Pad(b)
        x = torch.zeros(self.full_sz)
        eta_l1 = torch.zeros(self.full_sz)
        eta_tv = torch.zeros_like(self.Delta(x))
        tau = torch.zeros(self.full_sz)
        sigma = torch.zeros(self.full_sz)

        for i in range(self.iter):
            x_pre = x
            u_l1 = self.update_ui(x, eta_l1, mode = "l1")
            u_tv = self.update_ui(x, eta_tv, mode = "tv")

            w = self.update_w(x, tau) 
            x = self.update_x(b, u_l1, eta_l1, u_tv, eta_tv, w, tau)                      

            eta_l1 = self.update_eta(x, u_l1, eta_l1, mode="l1")
            eta_tv = self.update_eta(x, u_tv, eta_tv, mode="tv")
            tau    = self.update_tau(x, w, tau)           
            
            if self.autotune:
                primal_res_w = torch.norm(w-x)
                dual_res_w = torch.norm((x - x_pre)) * self.alpha
                self.alpha = self.update_param(self.alpha, primal_res_w, dual_res_w) 

                primal_res_l1 = torch.norm(u_l1-x)
                dual_res_l1 = torch.norm((x - x_pre))*self.gamma_l1
                self.gamma_l1 = self.update_param(self.gamma_l1, primal_res_l1, dual_res_l1)

                Dx = self.Delta(x)
                Dx_pre = self.Delta(x_pre)
                primal_res_tv = torch.sqrt(torch.norm(Dx[:,:,0] -u_tv[:,:,0]) ** 2 + torch.norm(Dx[:,:,1] -u_tv[:,:,1]) ** 2  ) 
                dual_res_tv = torch.sqrt(torch.norm(Dx[:,:,0] -Dx_pre[:,:,0]) ** 2 + torch.norm(Dx[:,:,1] -Dx_pre[:,:,1]) ** 2  ) * self.gamma_tv
                self.gamma_tv = self.update_param(self.gamma_tv, primal_res_tv, dual_res_tv)
                
                self.update_inv()
                
            if i % self.disp == 0:
                
                plt.imshow(self.Crop(x), cmap="gray")
                plt.colorbar()
                plt.show(block=False)
                plt.pause(2) # 显示1s
                plt.close()
        print("alpha: {} gamma_l1: {} gamma_tv:{}".format(self.alpha, self.gamma_l1, self.gamma_tv))
        return self.Crop(x)

if __name__ == '__main__':
    
    diffuser = "data/diffuser/im326.npy"
    lensed = "data/lensed/im326.npy"
    psf_file = "data/psf.tiff"
    compress_img = np.load(diffuser)
    h,w,ch = compress_img.shape
    recons = list()

    iter = 100
    for i in range(ch):
        admm = ADMM(psf_file, mode="l1_tv", iters = iter, senor_size =[w,h], rgb_idx = i, disp = iter, autotune=True)
        data = compress_img[:,:,i]
        data /= np.linalg.norm(data.ravel(),ord=2) #这种方式归一化后，均值在0.002左右
        recon = np.array(admm.forward(torch.tensor(data,dtype=torch.float)))
        recon /= np.max(recon)
        recon[recon<0] = 0
        recons.append(recon)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(np.array(recons).transpose(1,2,0))
    ax[0].set_title("Final reconstructed image")
    ax[1].imshow(np.load(lensed))
    ax[1].set_title("Lensed image")
    plt.show()
