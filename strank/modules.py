import torch
import torch.nn as nn
import pytorch_lightning as pl

from loss_functions import (
    MSELoss,
    NBLoss,
    PoissonLoss,
    PearsonLoss,
    STRankLoss,
    STRankLossList,
    RankingLoss,
)

_eps = 1e-6


class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = inputs["img_feat"]
        outputs = {"gene_pred": self.fc(x)}
        return outputs


class FFlayer(nn.Module):
    def __init__(
        self, in_features, out_features, mid_dim=256, layer_num=2, dropout_p=0.5
    ):
        super().__init__()
        layers = (
            [
                nn.Linear(in_features, mid_dim),
                nn.BatchNorm1d(mid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
            ]
            + [
                nn.Sequential(
                    nn.Linear(mid_dim, mid_dim),
                    nn.BatchNorm1d(mid_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_p),
                )
                for _ in range(layer_num - 1)
            ]
            + [nn.Linear(mid_dim, out_features)]
        )
        self.f = nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs["img_feat"]
        outputs = {"gene_pred": self.f(x)}
        return outputs


_model_dict = {"linear": LinearModel, "ff": FFlayer}

_loss_dict = {
    "mse": MSELoss,
    "stranklist": STRankLossList,
    "stranklist_v2": STRankLoss,
    "strankg": STRankLoss,
    "stranka": STRankLoss,
    "strankgfw": STRankLoss,
    "poisson": PoissonLoss,
    "nb": NBLoss,
    "pearsona": PearsonLoss,
    "ranking": RankingLoss,
}


class STPred(pl.LightningModule):
    def __init__(self, model_key, loss_key, model_params={}, loss_params={}):
        super().__init__()
        self.model = _model_dict[model_key](**model_params)
        self.loss = _loss_dict[loss_key](**loss_params)
        self.lr = 5e-5

    def forward(self, inputs):
        outputs = self.model(inputs)
        loss = self.loss(outputs, inputs)
        return loss

    def training_step(self, batch, batch_idx):
        loss_dict = self(batch)
        loss = sum(loss_dict.values())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict = self(batch)
        for key, val in loss_dict.items():
            self.log(f"val_{key}", val)
        loss = sum(loss_dict.values())
        out = self.model(batch)
        scc = (
            loss  # pearson_corrcoef(batch['exp'], out['gene_pred']).nan_to_num().mean()
        )
        self.log("val_scc", loss)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    input = {
        "count": torch.randint(0, 10000, (256, 2)),
        "sample_id": torch.randint(0, 5, (256,)),
    }
    output = {"gene_pred": torch.randn(256, 2)}
    criterion = STRankLoss()

    val = criterion(output, input)
