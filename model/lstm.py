import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Use the output from the last time step
        out = self.fc(out[:, -1, :])
        return out


    def train_model(self, train_loader, val_loader=None, lr=1e-3, epochs=50, device=None):
        writer = SummaryWriter(log_dir=os.path.join("runs", "LSTM_Finance_Experiment"))

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
                    torch.save(self.state_dict(), "best_model.pth")


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

    import matplotlib.pyplot as plt

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

            # 🚀 ارسال سری زمانی قیمت واقعی و پیش‌بینی‌شده به TensorBoard
            preds = outputs_tensor.numpy()
            reals = targets_tensor.numpy()

            for i, (real, pred) in enumerate(zip(reals, preds)):
                writer.add_scalars("Test/Price_Comparison", {
                    'Real': real,
                    'Predicted': pred
                }, global_step=i)

        return avg_test_loss, outputs_tensor, targets_tensor






