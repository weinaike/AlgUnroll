
import torch


def SoftThresh(val, thresh):
    # numpy automatically applies functions to each element of the array
    return torch.sign(val)*torch.maximum(torch.abs(val) - thresh, torch.tensor(0))



class RegularBlock(torch.nn.Module):
    def __init__(self, filter = 32,  kernel_size = 3):
        super(RegularBlock, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv1d(1, filter, kernel_size=kernel_size, padding = kernel_size//2), 
                                        torch.nn.ReLU(), 
                                        torch.nn.Conv1d(filter, 1, kernel_size=kernel_size, padding = kernel_size//2))
    def forward(self,x):
        x = torch.unsqueeze(x, 1)
        x = self.block(x)
        x = torch.squeeze(x)
        return x


class ResiualLayer(torch.nn.Module):
    def __init__(self, filter = 32,  kernel_size = 3):
        super(ResiualLayer, self).__init__()
    
    def forward(self,x):
        return x

class CausalSelfAttention(torch.nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = torch.nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = torch.nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        n = x.size(0)
        x = x.view(n, 1, -1)
        # print(x.size())
        query_projected = self.c_attn(x)
        # print(query_projected.size())

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        # print(query.size(), key.size(), value.size())
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        # print(query.size(), key.size(), value.size())
        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        # print(self.num_heads, head_dim)
        # print(y.size())
        y = y.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        y = y.view(n,-1)
        return y


# num_heads = 8
# heads_per_dim = 64
# embed_dimension = num_heads * heads_per_dim
# dtype = torch.float16
# model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=False, is_causal=True, dropout=0.1).to("cuda").to(dtype).eval()
# print(model)