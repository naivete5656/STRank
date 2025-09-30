import torch
import torch.nn as nn

from torch.distributions import NegativeBinomial
from torchmetrics.functional.regression import spearman_corrcoef, pearson_corrcoef

_eps = 1e-6


## functions
def group_perm(groups, *xs):
    """permute the tensor x according to the groups. The elements in the same group are permuted.


    Args:
        xs (tupple): tupple of tensors to be permuted
        groups (torch.tensor): group indices

    Returns:
        perm_x (torch.tensor): permuted tensor
    """
    uniq_groups = torch.unique(groups)
    group_idxs = [torch.where(groups == group)[0] for group in uniq_groups]
    perm_xs = [torch.zeros_like(x) for x in xs]
    perm_idxs = torch.zeros_like(groups)
    for group_idx in group_idxs:
        perm_idx = torch.randperm(len(group_idx))
        perm_idxs[group_idx] = group_idx[perm_idx]
    perm_xs = [x[perm_idxs] for x in xs]
    return perm_xs


def global_perm(groups, *xs):
    """permute the tensor x according to the groups. The elements in the same group are permuted.


    Args:
        xs (tupple): tupple of tensors to be permuted
        groups (torch.tensor): group indices

    Returns:
        perm_x (torch.tensor): permuted tensor
    """
    perm_idx = torch.randperm(len(groups))
    perm_xs = [x[perm_idx] for x in xs]
    return perm_xs


def calc_raw_lpi_norm(stack_count, stack_pred):
    raw_lpi = stack_pred + (stack_count.float().mean(dim=-1, keepdims=True) + _eps).log()
    return raw_lpi


def calc_raw_lpi_non_norm(stack_count, stack_pred):
    raw_lpi = stack_pred
    return raw_lpi


## Losses


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, outputs, inputs):
        loss_dict = {"mse": self.criterion(outputs["gene_pred"], inputs["exp"])}
        return loss_dict


class STRankLossList(torch.nn.Module):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None):
        super().__init__()
        self.lsoftmax = torch.nn.LogSoftmax(dim=-2)
        self.calc_raw_lpi = calc_raw_lpi_non_norm
        if normalize_effect:
            self.calc_raw_lpi = calc_raw_lpi_norm
        else:
            self.calc_raw_lpi = calc_raw_lpi_non_norm
        self.group = perm

    def forward(self, outputs, inputs):
        count = inputs["count"]
        pred = outputs["gene_pred"]
        groups = inputs["sample_id"]
        if self.group == "group":
            loss = []
            for gl_id in torch.unique(groups):
                idx = groups == gl_id
                pred_gl = pred[idx]
                count_gl = count[idx]
                loss.append(-self.lsoftmax(self.calc_raw_lpi(count_gl, pred_gl)) * count_gl)
        else:
            loss = -self.lsoftmax(self.calc_raw_lpi(count, pred)) * count
        loss_dict = {
            "rank_binom_loss": torch.cat(loss).mean(),
        }
        return loss_dict


class STRankLoss(torch.nn.Module):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, n_pair=2):
        super().__init__()
        # self.lsoftmax = torch.nn.LogSoftmax(dim=1)
        self.lsoftmax = torch.nn.LogSoftmax(dim=-2)
        self.n_pair = n_pair
        if normalize_effect:
            self.calc_raw_lpi = calc_raw_lpi_norm
        else:
            self.calc_raw_lpi = calc_raw_lpi_non_norm
        if perm == "group":
            self.perm = group_perm
        else:
            self.perm = global_perm
        if feature_weights is not None:
            self.register_buffer("feature_weights", torch.tensor(feature_weights).float())
        else:
            self.feature_weights = None

    def forward(self, outputs, inputs):
        count = inputs["count"]
        pred = outputs["gene_pred"]
        groups = inputs["sample_id"]
        idxs = torch.arange(len(count)).to(count.device)

        # make n_pair for each batch
        count_list = []
        pred_list = []
        for n in range(self.n_pair):
            perm_count, perm_pred, perm_idxs = self.perm(groups, count, pred, idxs)
            count_list.append(perm_count)
            pred_list.append(perm_pred)
        stack_count = torch.stack(count_list, dim=1)
        stack_pred = torch.stack(pred_list, dim=1)
        lpi = self.lsoftmax(self.calc_raw_lpi(stack_count, stack_pred))

        if self.feature_weights is not None:
            lpi = lpi * self.feature_weights
        # mask = (idxs != perm_idxs).float()
        # sample_wise_loss = - torch.mean(lpi * stack_count, dim=[1, 2])
        # loss_dict = {
        #     "rank_binom_loss": torch.sum(sample_wise_loss * mask)  / (mask.sum() + _eps)
        # }
        pi_prior = -(torch.exp(lpi) * lpi).mean()
        loss_dict = {
            "rank_binom_loss": torch.mean(-lpi * stack_count),
            "pi_prior": pi_prior,
        }
        return loss_dict


class PoissonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, inputs):
        count = inputs["count"]
        pred = outputs["gene_pred"] + torch.log(count.mean(dim=1, keepdims=True))
        loss_dict = {"poisson_loss": torch.mean(torch.exp(pred) - count * pred)}
        return loss_dict


class NBLoss(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self._theta = nn.Parameter(torch.zeros(feature_dim))

    def reset_parameters(self):
        nn.init.zeros_(self._theta)

    @property
    def theta(self):
        return torch.exp(self._theta)

    def forward(self, outputs, inputs):
        count = inputs["count"]
        pred = outputs["gene_pred"] + torch.log(count.mean(dim=1, keepdims=True))
        ld = torch.exp(pred)
        p = ld / (ld + self.theta)
        p_z = NegativeBinomial(self.theta, p)
        loss = -p_z.log_prob(count).mean()
        loss_dict = {"nb_loss": loss}
        return loss_dict


class RankingLoss(torch.nn.Module):
    """
    The inputs are nomalized by the total count.
    """

    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, n_pair=32):
        super().__init__()
        self.calc_raw_lpi = calc_raw_lpi_norm
        self.perm = group_perm
        self.n_pair = n_pair

        self.criterion = nn.MarginRankingLoss()

    def forward(self, outputs, inputs):
        count = inputs["exp"]
        pred = outputs["gene_pred"]
        groups = inputs["sample_id"]
        idxs = torch.arange(len(count)).to(count.device)

        loss = 0
        for n in range(self.n_pair):
            perm_count, perm_pred, perm_idxs = self.perm(groups, count, pred, idxs)
            loss += self.criterion(pred, perm_pred, torch.where(count > perm_count, 1, -1))
        loss_dict = {"ranking": loss / self.n_pair}
        return loss_dict


class PearsonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, inputs):
        exp = inputs["exp"]
        pred = outputs["gene_pred"]
        loss_dict = {"pearson_loss": -pearson_corrcoef(exp, pred).nan_to_num().mean()}
        return loss_dict
