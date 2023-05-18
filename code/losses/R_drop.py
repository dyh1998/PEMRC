from torch import nn
from torch.nn import functional as F


class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """

    def __init__(self):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')

    def forward(self, logits1, logits2, target, kl_weight=1):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """

        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + kl_weight * kl_loss
        return loss


class MRCRDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """

    def __init__(self):
        super(MRCRDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')

    def forward(self, start_logits1, start_logits2, end_logits1, end_logits2, start_target, end_target, kl_weight=1):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """
        sl1 = start_logits1.permute(0, 2, 1)
        sl2 = start_logits2.permute(0, 2, 1)
        el1 = end_logits1.permute(0, 2, 1)
        el2 = end_logits2.permute(0, 2, 1)
        start_ce_loss = (self.ce(sl1, start_target) + self.ce(sl2, start_target)) / 2
        end_ce_loss = (self.ce(el1, end_target) + self.ce(el2, end_target)) / 2
        # print("start ce loss:", start_ce_loss)
        start_kl_loss1 = self.kld(F.log_softmax(start_logits1, dim=-1), F.softmax(start_logits2, dim=-1)).sum(-1)
        start_kl_loss2 = self.kld(F.log_softmax(start_logits2, dim=-1), F.softmax(start_logits1, dim=-1)).sum(-1)
        end_kl_loss1 = self.kld(F.log_softmax(end_logits2, dim=-1), F.softmax(end_logits1, dim=-1)).sum(-1)
        end_kl_loss2 = self.kld(F.log_softmax(end_logits1, dim=-1), F.softmax(end_logits2, dim=-1)).sum(-1)
        # print("start kl loss:", start_kl_loss1)
        start_kl_loss = (start_kl_loss1 + start_kl_loss2) / 2
        end_kl_loss = (end_kl_loss1 + end_kl_loss2) / 2

        loss = start_ce_loss + end_ce_loss + kl_weight * (start_kl_loss + end_kl_loss)
        return loss
