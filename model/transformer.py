import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os


class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, output_size=1):
        super(TransformerTimeSeriesModel, self).__init__()

        self.model_dim = model_dim

        self.input_proj = nn.Linear(input_size, model_dim)
        self.positional_encoding = self._generate_positional_encoding(500, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(model_dim, output_size)

    def _generate_positional_encoding(self, max_len, model_dim):
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, model_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)

        pe = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pe

        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])  # Use last time step output
        return out

    def train_model(self, train_loader, val_loader=None, lr=1e-3, epochs=50, device=None):
        writer = SummaryWriter(log_dir=os.path.join("runs", "Transformer_Finance_Experiment"))

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            writer.add_scalar('Loss/Train', total_loss, epoch)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}")

            if val_loader:
                val_loss = self.validate(val_loader, criterion, device)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.state_dict(), "best_transformer_model.pth")
                    print("✅ Model saved")

        writer.close()

    def validate(self, val_loader, criterion, device):
        self.eval()
        total_val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = self(x)
                loss = criterion(output, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def test_model(self, test_loader, device=None, log_tensorboard=False, writer=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval()
        self.to(device)

        criterion = nn.MSELoss()
        total_test_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = self(x)
                loss = criterion(output, y)
                total_test_loss += loss.item()

                all_outputs.append(output.cpu())
                all_targets.append(y.cpu())

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"🧪 Test Loss: {avg_test_loss:.4f}")

        outputs_tensor = torch.cat(all_outputs, dim=0).squeeze()
        targets_tensor = torch.cat(all_targets, dim=0).squeeze()

        if log_tensorboard and writer:
            writer.add_scalar("Loss/Test", avg_test_loss)

            preds = outputs_tensor.numpy()
            reals = targets_tensor.numpy()

            for i, (real, pred) in enumerate(zip(reals, preds)):
                writer.add_scalars("Test/Price_Comparison", {
                    'Real': real,
                    'Predicted': pred
                }, global_step=i)

        return avg_test_loss, outputs_tensor, targets_tensor
