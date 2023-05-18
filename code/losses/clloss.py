import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CosineSimilarity, PairwiseDistance


# def kl_sim(ins1, ins2):
#     kld = nn.KLDivLoss(reduction='none')  # 这里不能使用mean，会报维度不匹配的错误，得用batchmean
#     kl_loss1 = kld(F.log_softmax(ins1, dim=-1), F.softmax(ins2, dim=-1)).sum(-1)
#     kl_loss2 = kld(F.log_softmax(ins2, dim=-1), F.softmax(ins1, dim=-1)).sum(-1)
#     kl_loss = (kl_loss1 + kl_loss2) / 2
#     return 0.1 / kl_loss


def cos_sim(ins1, ins2):
    sim = CosineSimilarity(dim=-1, eps=1e-6)
    return sim(ins1, ins2)


def euc_sim(ins1, ins2):
    sim = PairwiseDistance(p=2)
    return sim(ins1, ins2)
