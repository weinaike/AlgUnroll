import torch
import numpy as np
from model_block import SoftThresh
import PIL.Image as Image
import matplotlib.pyplot as plt
class ADMM(torch.nn.Module):
    def __init__(self, mode = "only_tv"):
        # 步长

        self.gamma_l1 = 1              # u_l1
        self.gamma_tv = 0.0001              # u_tv
        self.gamma_dwt = 1              # u_dwt
        self.alpha = 1                  # w 
        self.beta =  1                  # g(z)        
        self.s = 1.0                    # for g(z)
        ## 拉格朗日乘子
        self.lambda_l1 = 1.0e-6
        self.lambda_tv = 1.0e-6
        self.lambda_dwt = 1.0e-6
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
        im = Image.open("data/psf.tiff")
        im = im.resize([480,270])
        psf = np.array(im,dtype='float32')
        psf = psf[:,:, 0]
        psf /= np.linalg.norm(psf.ravel())
        psf = torch.tensor(psf)        
        self.sz = psf.size()
        self.full_sz = torch.Size([self.sz[0]*2, self.sz[1]*2])

        self.H_fft = torch.fft.fft2(torch.fft.ifftshift(self.Pad(psf)))

        self.ATA = (torch.abs(torch.conj(self.H_fft)*self.H_fft))

        self.DeltaTDelta = self.precompute_DeltaTDelta()

        self.inv_item = 1.0/(self.ATA + self.gamma_tv * self.DeltaTDelta + (self.alpha + self.gamma_l1))




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

    def forward(self, b):
        iter = 300

        b = self.Pad(b)

        x = torch.zeros_like(b)
        eta_l1 = torch.zeros_like(x)
        eta_tv = torch.zeros_like(self.Delta(x))
        tau = torch.zeros_like(x)


        for i in range(iter):
            u_l1 = self.update_ui(x, eta_l1, mode = "l1")
            u_tv = self.update_ui(x, eta_tv, mode = "tv")

            w = self.update_w(x, tau)
            x = self.update_x(b, u_l1, eta_l1, u_tv, eta_tv, w, tau)

            eta_l1 = self.update_eta(x, u_l1, eta_l1, mode="l1")
            eta_tv = self.update_eta(x, u_tv, eta_tv, mode="tv")

            tau = self.update_tau(x, w, tau)
            if i % 20 == 0:
                plt.imshow(self.Crop(x), cmap="gray")
                plt.colorbar()
                plt.show(block=False)
                plt.pause(2) # 显示1s
                plt.close()


        return self.Crop(x)

if __name__ == '__main__':
    admm = ADMM(mode="only_tv")
    diffuser = "data/diffuser/im43.npy"
    lensed = "data/lensed/im43.npy"
    x = np.load(diffuser)
    data = x[:,:,0]
    data /= np.linalg.norm(data.ravel())
    recon = admm.forward(torch.tensor(data,dtype=torch.float))
    
    # plt.rcParams['figure.figsize'] = (8.0, 4.0) 
    plt.figure()
    plt.imshow(recon,cmap="gray")
    plt.show()
