from diffusers import DDIMScheduler
from torch import nn
import torch.nn.functional as F
import torch

class ResConv1D(nn.Module):
    def __init__(self, dropout=0.1):
        # not change the input shape
        super(ResConv1D, self).__init__() 
        self.GN1 = nn.GroupNorm(8, 64)   
        self.Conv1 = nn.Conv1d(64, 64, 3, padding=1)
        self.GN2 = nn.GroupNorm(8, 64)
        self.EtLinear = nn.Linear(1, 128)   # can be modified
        self.EsemLinear = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout) 
        self.Conv2 = nn.Conv1d(64, 64, 3, padding=1)

    def forward(self, x, E_t, E_sem):
        x_res = x
        x = self.GN1(x)
        x = F.silu(x)
        x = self.Conv1(x)
        x = self.GN2(x)             # shape: (batch_size, 64, seq_length)
        x = x.permute(0, 2, 1)      # shape: (batch_size, seq_length, 64)

        E_t = self.EtLinear(F.silu(E_t.float())) # shape: (batch, 128)
        E_t_mul, E_t_add = torch.chunk(E_t, 2, dim=1)
        x = x.permute(1, 0, 2) 
        x = x * E_t_mul + E_t_add

        E_sem = E_sem.squeeze(-1)   # shape: (batch_size, 128)
        E_sem = self.EsemLinear(F.silu(E_sem)) # shape: (64)
        x = x * E_sem 
        x = x.permute(1, 2, 0)      # shape: (batch_size, 64, seq_length)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.Conv2(x)
        return x + x_res

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.entranceConv = nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3, padding=1)

        self.downList = nn.ModuleList([
            ResConv1D(),
            nn.MaxPool1d(kernel_size=2),
            ResConv1D(),
            nn.MaxPool1d(kernel_size=2),
            ResConv1D(),
            nn.MaxPool1d(kernel_size=2),
        ])

        self.bottleNeck = ResConv1D()

        self.upList = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            ResConv1D(),
            nn.Upsample(scale_factor=2),
            ResConv1D(),
            nn.Upsample(scale_factor=2),
            ResConv1D(),
        ])

        self.GN = nn.GroupNorm(8, 64)
        self.outProj = nn.Conv1d(64, 64, 3, padding=1)

    def forward(self, x, E_t, E_sem):
        # x shape: (9, seq_length)
        x = self.entranceConv(x)
        for layer in self.downList:
            if isinstance(layer, ResConv1D):
                x = layer(x, E_t, E_sem)
            else:
                x = layer(x)
        x = self.bottleNeck(x, E_t, E_sem)
        for layer in self.upList:
            if isinstance(layer, ResConv1D):
                x = layer(x, E_t, E_sem)
            else:
                x = layer(x)
        x = self.GN(x)
        x = F.silu(x)
        x = self.outProj(x)
        return x

if __name__ == "__main__":
    model = Unet()
    input_data = torch.randn(64, 9, 40)
    output = model(input_data, torch.randint(0, 1000, (64, 1)), torch.rand(64, 128, 1))
    print(output.shape) # torch.Size([64, 64, 40])