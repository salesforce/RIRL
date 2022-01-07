import torch
import torch.nn as nn


class MIClassifier(nn.Module):
    """For estimating Mutual Information"""
    def __init__(self, x_dim=1, h_dim=32, lr = 5e-3):
        super().__init__()
        
        self.x_dim = int(x_dim)
        
        self.logits = nn.Sequential(
            nn.Linear(self.x_dim+1, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2),
        )
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

    def forward(self, xy):
        return self.logits(xy)
        
    def log_odds_ratio(self, xy):
        log_probs = torch.log(self.softmax(self.logits(xy)) + 1e-5)
        return log_probs[:, 1] - log_probs[:, 0]
        
    def predict(self, xy):
        probs = self.softmax(self.logits(xy))
        return probs[:, 1]
    
    def get_loss(self, xy, target):
        loss = self.loss(self.logits(xy), target)
        return loss.mean()