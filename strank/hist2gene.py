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
from transformer import ViT

_eps = 1e-6


_loss_dict = {
    "mse": MSELoss,
    "stranklist": STRankLossList,
    "strankg": STRankLoss,
    "stranka": STRankLoss,
    "strankgfw": STRankLoss,
    "poisson": PoissonLoss,
    "nb": NBLoss,
    "pearsona": PearsonLoss,
    "ranking": RankingLoss,
}


class His2Gene(pl.LightningModule):
    def __init__(
        self,
        model_key,
        loss_key,
        model_params={},
        loss_params={},
        patch_size=112,
        dim=1024,
        n_pos=64,
        n_layers=4,
        dropout=0.1,
        n_genes=250,
    ):

        super().__init__()
        patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2 * dim, dropout=dropout, emb_dropout=dropout)

        self.gene_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_genes))

        self.loss = _loss_dict[loss_key](**loss_params)
        self.lr = 5e-5

    def forward(self, inputs):
        patches = self.patch_embedding(
            inputs["img_feat"].unsqueeze(0)
        )  # shape: (batch_size, num_patches, embedding_dim)
        centers_x = self.x_embed(inputs["coords"][:, 0].unsqueeze(0))
        centers_y = self.y_embed(inputs["coords"][:, 1].unsqueeze(0))
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)

        outputs = {"gene_pred": x[0]}

        loss = self.loss(outputs, inputs)
        return loss, outputs

    def training_step(self, batchs, batch_idx):
        loss_total = 0
        for batch in batchs:
            loss_dict, _ = self(batch)
            loss_total += sum(loss_dict.values())
        loss = loss_total / (len(batch) + 1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch[0]  # Assuming batch is a list of batches
        loss_dict, _ = self(batch)
        for key, val in loss_dict.items():
            self.log(f"val_{key}", val)
        loss = sum(loss_dict.values())
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
