# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class SSMLoss(nn.Module):
    def __init__(self):
        """ SampledSoftmaxCrossEntropyLoss
        :param num_negs: number of negative instances in bpr loss.
        """
        super(SSMLoss, self).__init__()

    def forward(self, pos_score, neg_score, num_neg_items=None):
        """
        :param pos_score: predicted values of shape (batch_size, ) 
        :param neg_score: predicted values of shape (batch_size, num_negs)
        """
        scores = torch.cat((pos_score.unsqueeze(1), neg_score), dim=1)
        probs = F.softmax(scores, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss

# TODO: models using CCL must be modified
class CCLoss(nn.Module):
    def __init__(self, negative_weight=None):
        """
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CCLoss, self).__init__()
        self._negative_weight = negative_weight

    def forward(self, pos_score, neg_score, margin, num_neg_items=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param pos_score: predicted values of shape (batch_size, ) 
        :param neg_score: predicted values of shape (batch_size, num_negs)
        """
        pos_loss = torch.relu(1 - pos_score) # TODO relu 빼보기
        neg_loss = torch.relu(neg_score - margin)
        if self._negative_weight:
            if num_neg_items is None:
                loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
            else:    
                loss = pos_loss + neg_loss.sum(dim=-1) / num_neg_items * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()


class DualCCLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(DualCCLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight

    def forward(self, pos_score, neg_score):
        """
        :param pos_score: predicted values of shape (batch_size, ) 
        :param neg_score: predicted values of shape (batch_size, num_negs)
        """
        pos_loss = torch.relu(1 - pos_score) # TODO relu 빼보기
        neg_loss = torch.relu(neg_score - self._margin)
        if self._negative_weight:
            loss = pos_loss.mean(dim=-1) + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss.sum(dim=-1) + neg_loss.sum(dim=-1)
        return loss.mean()


class RegLoss(nn.Module):
    """ RegLoss, L2 regularization on model parameters

    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """ EmbMarginLoss, regularization on embeddings
    """

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
