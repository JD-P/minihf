import torch.nn as nn

class RewardHead(nn.Module):
    def __init__(self):
        super(RewardHead, self).__init__()
        self.linear1 = nn.Linear(2048, 2048)
        self.gaussian1 = nn.GELU()
        self.linear2 = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gaussian1(x)
        x = self.linear2(x)
        return x
