import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 768
        self.D = 192
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1)
            # nn.Sigmoid()
        )
        self.ReLU = nn.ReLU()
        self.dropout1 = nn.Dropout(0.10)
        self.dropout2 = nn.Dropout(0.10)

    def forward(self, x):
        x = x.squeeze(0)
        x = self.dropout1(x)
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x)  # KxL

        M = self.dropout2(M)
        Y_prob = self.classifier(M)
        return [Y_prob, A]