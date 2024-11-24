import torch
import torch.nn.functional as F
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, in_feature=3, seq_length=40, hidden_dim=16):
        super(STGCN, self).__init__()
        self.temporalMat = nn.Parameter(torch.randn(seq_length, seq_length))
        self.featureMat = nn.Parameter(torch.randn(in_feature, hidden_dim))
        self.spatialMat = nn.Parameter(torch.randn(3, 3))
        self.in_feature = in_feature
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = x.view(-1, self.in_feature, 3, self.seq_length) 
        x = torch.einsum('bfsl,lm->bfsm', x, self.temporalMat)
        x = torch.einsum('bfsl,fm->bmsl', x, self.featureMat)
        x = torch.einsum('bfsl,sm->bfml', x, self.spatialMat)
        x = x.reshape(-1, self.hidden_dim * 3, self.seq_length)
        return x      
    
class resGCN(nn.Module):
    def __init__(self, in_feature=16, dropout=0.1, seq_length=40):
        super(resGCN, self).__init__()
        self.stgcn = STGCN(in_feature=in_feature, seq_length=seq_length)
        self.LN = nn.LayerNorm(3 * in_feature) # to be modified
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_res = x # shape: (batch_size, 3 * hidden_dim, seq_length)
        x = self.stgcn(x)
        x = x.permute(0, 2, 1)
        x = self.LN(x)
        x = x.permute(0, 2, 1)
        x = F.tanh(x)
        x = self.dropout(x)
        return x + x_res

class semanticEncoder(nn.Module):
    def __init__(self, in_feature=3, hidden_dim=16, dropout=0.1, seq_length=40):
        super(semanticEncoder, self).__init__()
        self.entranceSTGCN = STGCN(in_feature=in_feature)
        self.resGCNSeq = nn.Sequential(
            resGCN(in_feature=hidden_dim, dropout=dropout),
            nn.AvgPool1d(kernel_size=2),
            resGCN(in_feature=hidden_dim, dropout=dropout, seq_length=seq_length//2),
            nn.AvgPool1d(kernel_size=2),
            resGCN(in_feature=hidden_dim, dropout=dropout, seq_length=seq_length//4),
        )
        self.out_proj = nn.Sequential(
            nn.AvgPool1d(kernel_size=seq_length//4, stride=seq_length//4),
            nn.Conv1d(in_channels=3 * hidden_dim, out_channels=128, kernel_size=1),
        )

    def forward(self, x):
        x = self.entranceSTGCN(x)
        x = self.resGCNSeq(x)
        semantic_representation = self.out_proj(x)
        return x, semantic_representation

# Example usage
if __name__ == "__main__":
    model = semanticEncoder()
    input_data = torch.randn(64, 9, 40)
    output_data = model(input_data)
    print(output_data[0].shape, output_data[1].shape)  # Should print torch.Size([batch, 48, 10])

    # [batch, 128, 1]