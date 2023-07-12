import torch
import numpy as np
from model_block import SoftThresh
import PIL.Image as Image
import matplotlib.pyplot as plt
import math

def closest_power_of_two(n):
    power = math.floor(math.log2(n)) + 1
    result = 2 ** power
    return result

class ADMM(torch.nn.Module):
    def __init__(self, psf_file, mode = "tv", iters = 100, senor_size =[480,270], rgb_idx = 0, disp = 20, autotune = False ):
        # 步长
        self.iter = iters
        self.disp = disp
        self.gamma_tv = 5e-4               # u_tv
        self.alpha = 5e-4                 # w 
        self.delta = 5e-5                  # M
  
        ## 拉格朗日乘子
        self.lambda_tv = 1.0e-4            # u_tv

        im = Image.open(psf_file)
        im = im.resize(senor_size) 
        psf = np.array(im,dtype='float32')
        h, w, c = psf.shape
        psf /= np.linalg.norm(psf.ravel())
        psf = torch.tensor(psf).permute(2,0,1)
        

        self.c = c
        self.sz = torch.Size([h, w])
        self.full_sz = torch.Size([closest_power_of_two(self.sz[0]*2), closest_power_of_two(self.sz[1]*2)])
        # self.full_sz = torch.Size([self.sz[0]*2, self.sz[1]*2])

        self.H_fft = torch.fft.fft2(torch.fft.ifftshift(self.Pad(psf), dim=(-2, -1)))
        self.MTM = (torch.abs(torch.conj(self.H_fft)*self.H_fft)) #这个需要再探讨
        self.DeltaTDelta = self.precompute_DeltaTDelta()

        self.inv_item = 1.0/(self.delta * self.MTM + self.gamma_tv * self.DeltaTDelta + self.alpha)

        self.autotune = autotune

        self.res_tol = 1.5
        self.mu_inc  = 1.2
        self.mu_dec  = 1.2


    def update_inv(self):
        self.inv_item = 1.0/(self.delta * self.MTM + self.gamma_tv * self.DeltaTDelta + self.alpha )

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
        return PsiTPsi.repeat(self.c,1,1)
    
    def Delta(self, img):
        return torch.stack((torch.roll(img,1,dims=1) - img, torch.roll(img,1,dims=2) - img), dim=3)

    def DeltaT(self, x_diff):
        vec1 = x_diff[:,:,:,0]
        vec2 = x_diff[:,:,:,1]
        diff1 = torch.roll(vec1, -1, dims=1) - vec1
        diff2 = torch.roll(vec2, -1, dims=2) - vec2
        return diff1 + diff2    
    
    def Crop(self,M):
        # Image stored as matrix (row-column rather than x-y)
        top = (self.full_sz[0] - self.sz[0])//2
        bottom = (self.full_sz[0] + self.sz[0])//2
        left = (self.full_sz[1] - self.sz[1])//2
        right = (self.full_sz[1] + self.sz[1])//2
        return M[:,top:bottom,left:right]

    def CTC(self,x):
        return self.Pad(self.Crop(x))

    def Pad(self,b):
        v_pad = (self.full_sz[0] -  self.sz[0])//2
        h_pad = (self.full_sz[1] -  self.sz[1])//2
        return torch.nn.functional.pad(b,(h_pad,h_pad,v_pad,v_pad),"constant",0)
    
    def PSF(self, x):
        Mx = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.ifftshift(x,dim=(-2,-1))) * self.H_fft), dim=(-2,-1)))
        return Mx
    
    def conjPSF(self,x):        
        MTx = torch.real(torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2,-1))) * torch.conj(self.H_fft)),dim=(-2,-1)))
        return MTx

    def update_ui(self, x, eta, mode = "tv"):   
        val = self.Delta(x) + eta / self.gamma_tv
        thresh =  self.lambda_tv / self.gamma_tv
        return SoftThresh(val, thresh)

    
    def update_w(self, x, tau):
        val = x + tau/ self.alpha
        return torch.maximum(val,torch.tensor(0))
    

    def update_x(self, b, u_tv, eta_tv, w, tau, v, theta):
        resiual = self.conjPSF(self.delta * v - theta)
        resiual +=  self.alpha * w - tau        
        resiual += self.DeltaT(self.gamma_tv * u_tv - eta_tv)
        freq_space_result = torch.fft.fft2(torch.fft.ifftshift(resiual,  dim=(-2,-1)))
        x =  torch.real(torch.fft.fftshift(torch.fft.ifft2( self.inv_item *freq_space_result), dim=(-2,-1)))
        return x
        

    def update_v(self,x, b, theta):
        v = b + theta + self.delta * self.PSF(x)
        inv = 1/(self.Pad(self.Crop(torch.ones_like(v))) + self.delta)
        v = torch.mul(inv, v)
        return v


    def update_eta(self, x, u, eta, mode = "tv"):
        eta_n = eta + torch.mul(self.gamma_tv , self.Delta(x) - u)
        return eta_n


    def update_tau(self, x, w, tau):
        tau += torch.mul(self.alpha, (x - w))
        return tau
   
    def update_theta(self, x, v, theta):
        theta += torch.mul(self.delta, (self.PSF(x) - v))
        return theta

    def forward(self, b):

        b = self.Pad(b)
        x = torch.zeros_like(b)
        eta_tv = torch.zeros_like(self.Delta(x))
        tau = torch.zeros_like(x)
        theta = torch.zeros_like(x)

        for i in range(self.iter):
            x_pre = x

            u_tv = self.update_ui(x, eta_tv, mode = "tv")
            v = self.update_v(x, b, theta)     
            w = self.update_w(x, tau) 

            x = self.update_x(b, u_tv, eta_tv, w, tau, v, theta)    

            eta_tv = self.update_eta(x, u_tv, eta_tv, mode="tv")
            theta  = self.update_theta(x, v, theta)
            tau    = self.update_tau(x, w, tau)   
            
            if self.autotune:
                primal_res_w = torch.norm(w-x)
                dual_res_w = torch.norm((x - x_pre)) * self.alpha
                self.alpha = self.update_param(self.alpha, primal_res_w, dual_res_w) 

                Dx = self.Delta(x)
                Dx_pre = self.Delta(x_pre)
                primal_res_tv = torch.sqrt(torch.norm(Dx[:,:,0] -u_tv[:,:,0]) ** 2 + torch.norm(Dx[:,:,1] -u_tv[:,:,1]) ** 2  ) 
                dual_res_tv = torch.sqrt(torch.norm(Dx[:,:,0] -Dx_pre[:,:,0]) ** 2 + torch.norm(Dx[:,:,1] -Dx_pre[:,:,1]) ** 2  ) * self.gamma_tv
                self.gamma_tv = self.update_param(self.gamma_tv, primal_res_tv, dual_res_tv)


                Ax = self.PSF(x)
                Ax_pre= self.PSF(x_pre)
                primal_res_v = torch.norm(Ax-v)
                dual_res_v = torch.norm((Ax - Ax_pre)) * self.delta
                self.delta = self.update_param(self.delta, primal_res_v, dual_res_v) 
                
                self.update_inv()
                
            if i % self.disp == 0:
                x_out = self.Crop(x).permute(1,2,0)          
                x_out = x_out / torch.max(x_out)
                x_out[x_out<0]=0
                plt.imshow(x_out, cmap="gray")
                plt.show(block=False)
                plt.pause(2) # 显示1s
                plt.close()
        print("alpha: {} gamma_tv:{} delta:{}".format(self.alpha, self.gamma_tv, self.delta))
        return self.Crop(x)
    
if __name__ == '__main__':
    
    diffuser = "data/diffuser_images/im326.npy"
    lensed = "data/ground_truth_lensed/im326.npy"
    psf_file = "data/psf.tiff"
    compress_img = np.load(diffuser)
    h,w,ch = compress_img.shape

    iter = 100
    admm = ADMM(psf_file, mode="tv", iters = iter, senor_size =[w,h], disp = 10, autotune=True)

    compress_img /= np.linalg.norm(compress_img.ravel(),ord=2) #这种方式归一化后，均值在0.002左右
    recon = np.array(admm.forward(torch.tensor(compress_img,dtype=torch.float).permute(2,0,1)))
    recon /= np.max(recon)
    recon[recon<0] = 0

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(np.array(recon).transpose(1,2,0))
    ax[0].set_title("Final reconstructed image")
    ax[1].imshow(np.load(lensed))
    ax[1].set_title("Lensed image")
    plt.show()
