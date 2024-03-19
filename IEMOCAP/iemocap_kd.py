import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))
        
class Logit_Loss(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=2.0):
        super(Logit_Loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, z_s, z_t):
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

class Feature_Loss(nn.Module):
    def __init__(self, temp=1.0):
        super(Feature_Loss, self).__init__()
        self.t = temp
    def forward(self, other_embd, text_embd):
        text_embd = F.normalize(text_embd, p=2, dim=1)
        other_embd = F.normalize(other_embd, p=2, dim=1)
        target = torch.matmul(text_embd, text_embd.transpose(0,1))
        x = torch.matmul(text_embd, other_embd.transpose(0,1))
        log_q = torch.log_softmax(x / self.t, dim=1)
        p = torch.softmax(target / self.t, dim=1)
        return F.kl_div(log_q, p, reduction='batchmean')