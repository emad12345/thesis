import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, model_dim=64, num_heads=4, num_layers=2,
                 dropout=0.1, output_size=1, task='regression', num_classes=None):
        super(TransformerTimeSeriesModel, self).__init__()

        self.model_dim = model_dim
        self.task = task
        self.num_classes = num_classes if task == 'classification' else None

        self.input_proj = nn.Linear(input_size, model_dim)
        self.positional_encoding = self._generate_positional_encoding(500, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(model_dim, output_size if task == 'regression' else num_classes)

    def _generate_positional_encoding(self, max_len, model_dim):
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(np.array(1000), dtype=torch.float32)) / model_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.input_proj(x)
        pe = self.positional_encoding[:, :seq_len, :].to(x.device)
        x = x + pe

        x = self.transformer_encoder(x)
        out = self.fc(x[:, -1, :])  # آخرین تایم‌استپ

        if self.task == 'classification':
            if self.num_classes == 1:
                return torch.sigmoid(out)  # باینری
            else:
                return out  # چندکلاسه – loss خودش softmax می‌زنه
        else:
            return out  # رگرسیون

    def _get_loss_fn(self):
        if self.task == 'regression':
            return nn.MSELoss()
        elif self.task == 'classification':
            if self.num_classes == 1:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()
        else:
            raise ValueError("Invalid task type. Choose 'regression' or 'classification'.")

    def train_model(self, train_loader, val_loader=None, lr=1e-3, epochs=50, device=None):
        writer = SummaryWriter(log_dir=os.path.join("runs", f"Transformer_{self.task}_Experiment"))

        # هایپرپارامترها برای TensorBoard
        hparams = {
            'input_size': self.input_proj.in_features,
            'model_dim': self.model_dim,
            'num_heads': self.transformer_encoder.layers[0].self_attn.num_heads,
            'num_layers': len(self.transformer_encoder.layers),
            'dropout': self.transformer_encoder.layers[0].dropout.p,
            'output_size': self.fc.out_features,
            'task': self.task
        }

        final_metrics = {
            'hparam/final_train_loss': 0.0,
            'hparam/final_val_loss': 0.0
        }

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        criterion = self._get_loss_fn()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = float('inf')

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                x, y = x.to(device), y.to(device)

                if self.task == 'classification' and self.num_classes > 1:
                    y = y.long().squeeze()

                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            writer.add_scalar('Loss/Train', total_loss, epoch)
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / len(train_loader):.4f}")

            if val_loader:
                val_loss = self.validate(val_loader, criterion, device)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.state_dict(), f"best_transformer_{self.task}.pth")
                    print("✅ Model saved")

        final_metrics['hparam/final_train_loss'] = total_loss / len(train_loader)
        if val_loader:
            final_metrics['hparam/final_val_loss'] = best_loss

        writer.add_hparams(hparams, final_metrics)
        writer.close()

    def validate(self, val_loader, criterion, device):
        self.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                if self.task == 'classification' and self.num_classes > 1:
                    y = y.long().squeeze()

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

        criterion = self._get_loss_fn()
        total_test_loss = 0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                if self.task == 'classification' and self.num_classes > 1:
                    y = y.long().squeeze()

                output = self(x)
                loss = criterion(output, y)
                total_test_loss += loss.item()

                all_outputs.append(output.cpu())
                all_targets.append(y.cpu())

        outputs_tensor = torch.cat(all_outputs, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        avg_test_loss = total_test_loss / len(test_loader)

        print(f"🧪 Test Loss: {avg_test_loss:.4f}")

        if log_tensorboard and writer:
            writer.add_scalar("Loss/Test", avg_test_loss)

        if self.task == 'classification':
            y_true = targets_tensor.numpy()

            if self.num_classes == 1:
                y_prob = outputs_tensor.numpy()
                y_pred = (y_prob > 0.5).astype(int)
            else:
                y_prob = outputs_tensor.numpy()
                y_pred = y_prob.argmax(axis=1)

            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            print(f"✅ Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

            if log_tensorboard and writer:
                writer.add_scalar("Metrics/Test_Accuracy", acc)
                writer.add_scalar("Metrics/Test_Precision", prec)
                writer.add_scalar("Metrics/Test_Recall", rec)
                writer.add_scalar("Metrics/Test_F1", f1)

                # 🔵 Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf)
                image = transforms.ToTensor()(image)
                writer.add_image("Confusion_Matrix", image, dataformats='CHW')
                buf.close()
                plt.close(fig)

        return avg_test_loss, outputs_tensor, targets_tensor

