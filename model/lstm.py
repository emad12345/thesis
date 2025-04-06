import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss/len(train_loader):.4f}")

            if val_loader:
                val_loss = self.validate(val_loader, criterion, device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.state_dict(), "best_model.pth")
                    print("✅ Model saved")

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
