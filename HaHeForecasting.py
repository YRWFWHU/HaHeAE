from torch import nn
import torch.nn.functional as F
import torch

class HaHeForecasting(nn.Module):
    def __init__(self, seq_length=10, prediction_length=3, features=16):
        super(HaHeForecasting, self).__init__()
        self.features = features

        self.hand_conv1 = nn.Conv1d(in_channels=2 * features, out_channels=64, kernel_size=3, padding=1)
        self.LN_hand = nn.LayerNorm(64)
        self.hand_conv2 = nn.Conv1d(in_channels=64, out_channels=6, kernel_size=1)
        self.proj_hand = nn.Linear(seq_length, prediction_length)

        self.head_conv1 = nn.Conv1d(in_channels=features, out_channels=32, kernel_size=3, padding=1)
        self.LN_head = nn.LayerNorm(32)
        self.head_conv2 = nn.Conv1d(in_channels=32, out_channels=3, kernel_size=1)
        self.proj_head = nn.Linear(seq_length, prediction_length)

    def forward(self, x):
        # x shape: (batch_size, 16 * head_left_right, 10)
        head = self.head_conv1(x[:, :self.features, :])
        head = head.permute(0, 2, 1)
        head = self.LN_head(head)
        head = F.tanh(head)
        head = head.permute(0, 2, 1)
        head = self.head_conv2(head)
        head = F.tanh(head)
        head = self.proj_head(head)

        hand = self.hand_conv1(x[:, self.features:, :])
        hand = hand.permute(0, 2, 1)
        hand = self.LN_hand(hand)
        hand = F.tanh(hand)
        hand = hand.permute(0, 2, 1)
        hand = self.hand_conv2(hand)
        hand = F.tanh(hand)
        hand = self.proj_hand(hand)
        return torch.cat((head, hand), dim=1)
    
if __name__ == "__main__":
    model = HaHeForecasting()
    input_data = torch.randn(64, 48, 10)
    output = model(input_data)
    print(output.shape) # torch.Size([batch_size, 9, 3])
