
import torch


def SoftThresh(val, thresh):
    # numpy automatically applies functions to each element of the array
    return torch.sign(val)*torch.maximum(torch.abs(val) - thresh, torch.tensor(0))



class RegularBlock(torch.nn.Module):
    def __init__(self, filter = 32,  kernel_size = 3):
        super(RegularBlock, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(1, filter, kernel_size=kernel_size, padding = kernel_size//2), 
                                        torch.nn.ReLU(), 
                                        torch.nn.Conv2d(filter, 1, kernel_size=kernel_size, padding = kernel_size//2))
    def forward(self,x):        
        x = self.block(x)
        return x


