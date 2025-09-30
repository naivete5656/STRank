from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import spearman_corrcoef

from .loss_functions import (
    MSELoss,
    STRankLoss,
    GroupPearson,
    PearsonLoss,
    STRankLossPair,
    STRankLossK,
    STRankLoss2,
    STRankLoss4,
    STRankLoss8,
    STRankLoss16,
    STRankLoss32,
    STRankLoss64,
    STRankLoss128,
    STRankLossStable,
    RankingLoss,
    NBLoss,
    PoissonLoss,
)


# Example neural network for regression
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16):
        super(SimpleRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


class CustomDataset(Dataset):
    def __init__(self, x, y, gl=None):
        self.x = x
        self.y = y
        self.gl = gl  # gl_trainはオプショナル（テストデータにはないため）

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.gl is not None:
            return {"x": self.x[idx], "y": self.y[idx], "gl": self.gl[idx]}
        else:
            return {"x": self.x[idx], "y": self.y[idx]}


def train_model(
    dataset,
    wandb,
    loss_type="MSELoss",
    epochs=1000,
    learning_rate=0.01,
    best_weight_path="best_model.pth",
):
    """
    Train a neural network on the given data

    Parameters:
    - X: Input features
    - y: Target values
    - test_size: Proportion of data to use for testing
    - epochs: Number of training epochs
    - learning_rate: Learning rate for optimizer

    Returns:
    - Trained model
    - Training history
    """
    # Convert to PyTorch tensors
    x_train = torch.FloatTensor(dataset["x_train"])
    y_train = torch.FloatTensor(dataset["y_train"].reshape(-1, 1))
    gl_train = torch.FloatTensor(dataset["gl_train"])

    x_val = torch.FloatTensor(dataset["x_val"].reshape(-1, 1))
    y_val = torch.FloatTensor(dataset["y_val"].reshape(-1, 1))

    x_test = torch.FloatTensor(dataset["x_test"].reshape(-1, 1))
    y_test = torch.FloatTensor(dataset["y_test"].reshape(-1, 1))

    # トレーニングデータセットの作成
    train_dataset = CustomDataset(x_train, y_train, gl_train)

    # DataLoaderの作成
    batch_size = 256  # バッチサイズは必要に応じて調整してください
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # トレーニングデータはシャッフルする
        num_workers=0,  # 必要に応じて並列処理ワーカー数を調整
    )

    # # テストデータセットの作成
    # test_dataset = CustomDataset(x_test, y_test)
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,  # テストデータはシャッフルしない
    #     num_workers=0,
    # )

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleRegressor()
    model = model.to(device)

    criterion = eval(loss_type)()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001)

    best_val_scc = -1
    patience = 200
    counter = 0
    # Training loop
    train_losses = []
    for epoch in tqdm(range(epochs)):
        model.train()

        for idx, data in enumerate(train_loader):
            # Training phase
            optimizer.zero_grad()
            y_pred = model(data["x"].reshape(-1, 1).to(device))
            loss = criterion(
                y_pred, data["y"].reshape(-1, 1).to(device), data["gl"].to(device)
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            wandb.log({"train_loss": loss.item()})

        scheduler.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val.to(device))
            scc = spearman_corrcoef(y_val[:, 0].to(device), y_val_pred[:, 0])
            wandb.log({"test_scc": scc.item()})
        if best_val_scc < scc:
            counter = 0
            best_val_scc = scc
            # Save the model
            torch.save(model.state_dict(), best_weight_path)
            wandb.log({"best_val_scc": best_val_scc.item()})
        else:
            counter += 1
        if counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # # Print progress
        # if epoch % 20 == 0:
        #     print(
        #         # f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Test Loss = {test_loss.item():.4f}"
        #         f"Epoch {epoch}: Train Loss = {loss.item():.4f}"
        #     )

    # Test phase
    model.load_state_dict(torch.load(best_weight_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test.to(device))
        if loss_type == "MSELoss":
            y_test_pred = torch.exp(y_test_pred) - 1
    return y_test_pred.cpu().detach().numpy()
