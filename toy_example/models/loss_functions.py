import torch
import torch.nn as nn
from torchmetrics.functional.regression import pearson_corrcoef
from torch.distributions import NegativeBinomial

_eps = 1e-6


def calc_raw_lpi_norm(stack_count, stack_pred):
    raw_lpi = (
        stack_pred + (stack_count.float().mean(dim=-1, keepdims=True) + _eps).log()
    )
    return raw_lpi


def calc_raw_lpi_non_norm(stack_count, stack_pred):
    raw_lpi = stack_pred
    return raw_lpi


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
    perm_idxs = torch.zeros_like(groups, dtype=torch.long)
    for group_idx in group_idxs:
        perm_idx = torch.randperm(len(group_idx))
        perm_idxs[group_idx] = group_idx[perm_idx]
    perm_xs = [x[perm_idxs] for x in xs]
    return perm_xs

def stable_logsoftmax(raw_lpi):
    max_val = raw_lpi.max(dim=-2, keepdim=True).values
    raw_lpi_shifted = raw_lpi - max_val
    log_probs = raw_lpi_shifted - torch.log(torch.sum(torch.exp(raw_lpi_shifted), dim=-2, keepdim=True))
    return log_probs

class STRankLoss(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.lsoftmax = torch.nn.LogSoftmax(dim=-2)
        self.calc_raw_lpi = calc_raw_lpi_non_norm

    def forward(self, pred, count, groups):
        loss = []
        for gl_id in torch.unique(groups):
            idx = groups == gl_id
            pred_gl = pred[idx]
            count_gl = count[idx]
            loss.append(-self.lsoftmax(self.calc_raw_lpi(count_gl, pred_gl)) * count_gl)
        return torch.cat(loss).mean()
    
class STRankLossStable(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.lsoftmax = torch.nn.LogSoftmax(dim=-2)
        self.calc_raw_lpi = calc_raw_lpi_non_norm

    def forward(self, pred, count, groups):
        loss = []
        for gl_id in torch.unique(groups):
            idx = groups == gl_id
            pred_gl = pred[idx]
            count_gl = count[idx]
            raw_lpi = self.calc_raw_lpi(count_gl, pred_gl)
            log_probs = stable_logsoftmax(raw_lpi)
            nll = -log_probs * count_gl
            loss.append(nll)
        return torch.cat(loss).mean()


class STRankLossPair(torch.nn.Module):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=2):
        super().__init__()
        self.lsoftmax = torch.nn.LogSoftmax(dim=-2)
        # self.calc_raw_lpi = self.calc_raw_lpi_non_norm
        self.calc_raw_lpi = calc_raw_lpi_non_norm
        self.perm = group_perm
        self.k = k

        if feature_weights is not None:
            self.register_buffer(
                "feature_weights", torch.tensor(feature_weights).float()
            )
        else:
            self.feature_weights = None

    def forward(self, pred, count, groups):
        idxs = torch.arange(len(count)).to(count.device)

        # make n_pair for each batch
        count_list = []
        pred_list = []
        for n in range(self.k):
            perm_count, perm_pred, perm_idxs = self.perm(groups, count, pred, idxs)
            count_list.append(perm_count)
            pred_list.append(perm_pred)
        stack_count = torch.stack(count_list, dim=1)
        stack_pred = torch.stack(pred_list, dim=1)
        lpi = self.lsoftmax(self.calc_raw_lpi(stack_count, stack_pred))

        return torch.mean(-lpi * stack_count)


class STRankLossK(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=16):
        super().__init__(
            normalize_effect=True, perm="group", feature_weights=None, k=16
        )


class STRankLoss2(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=2):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLoss4(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=4):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLoss8(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=8):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLoss16(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=16):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLoss32(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=32):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLoss64(STRankLossPair):
    def __init__(self, normalize_effect=True, perm="group", feature_weights=None, k=64):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLoss128(STRankLossPair):
    def __init__(
        self, normalize_effect=True, perm="group", feature_weights=None, k=128
    ):
        super().__init__(
            normalize_effect=normalize_effect,
            perm=perm,
            feature_weights=feature_weights,
            k=k,
        )


class STRankLossReg(STRankLoss):
    def forward(self, pred, count, groups):
        idxs = torch.arange(len(count)).to(count.device)
        perm_count, perm_pred, perm_idxs = self.perm(groups, count, pred, idxs)
        stack_count = torch.stack([count, perm_count], dim=1)
        stack_pred = torch.stack([pred, perm_pred], dim=1)
        lpi = self.lsoftmax(self.calc_raw_lpi(stack_count, stack_pred))
        pi_prior = -(torch.exp(lpi) * lpi).mean()
        return torch.mean(-lpi * stack_count) + pi_prior


class RankingLoss(torch.nn.Module):
    def __init__(
        self, normalize_effect=True, perm="group", feature_weights=None, n_pair=32
    ):
        super().__init__()
        self.calc_raw_lpi = calc_raw_lpi_norm
        self.perm = group_perm
        self.n_pair = n_pair

        self.criterion = nn.MarginRankingLoss()

    def forward(self, pred, count, groups):
        idxs = torch.arange(len(count)).to(count.device)

        loss = 0
        for n in range(self.n_pair):
            perm_count, perm_pred, perm_idxs = self.perm(groups, count, pred, idxs)
            loss += self.criterion(
                pred, perm_pred, torch.where(count > perm_count, 1, -1)
            )
        return loss / self.n_pair


class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, count, groups=None):
        count = torch.log(1 + count)
        loss = self.loss(count, pred)
        return loss


class PoissonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, count, groups=None):
        pred = pred
        # + torch.log(count.mean(dim=1, keepdims=True) + _eps)
        return torch.mean(torch.exp(pred) - count * pred)


class NBLoss(torch.nn.Module):
    def __init__(self, feature_dim=1):
        super().__init__()
        self._theta = nn.Parameter(torch.zeros(feature_dim))

    def reset_parameters(self):
        nn.init.zeros_(self._theta)

    @property
    def theta(self):
        return torch.exp(self._theta)

    def forward(self, pred, count, groups=None):
        pred = pred
        # + torch.log(count.mean(dim=1, keepdims=True))
        ld = torch.exp(pred)

        p = ld / (ld + self.theta.to(pred.device))
        p_z = NegativeBinomial(self.theta, p)
        loss = -p_z.log_prob(count).mean()
        return loss


class PearsonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, exp, groups=None):
        corr = -pearson_corrcoef(exp, pred)
        if corr.isnan():
            corr = torch.tensor(0)
        return corr.mean()


class GroupPearson(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, exp, group=None):
        corr_list = []
        size_list = []
        for group_id in torch.unique(group):
            idx = group == group_id

            corr = pearson_corrcoef(exp[idx], pred[idx])
            size = idx.sum()
            if corr.isnan():
                corr = torch.tensor(0)
            corr_list.append(corr)
            size_list.append(size)

        corr_pearson = (
            torch.stack(corr_list) * torch.stack(size_list)
        ).sum() / torch.tensor(size_list).sum()
        return -corr_pearson
