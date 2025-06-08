"""

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.modules import ModelCore
from ...nn.modules.loss import Criterion
from ...nn.modules.timesnet import BackboneTimesNet
from ...nn.modules.transformer.embedding import DataEmbedding


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads , seq_length):
        super(CustomMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear layers for Q, K, V
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        # Final linear layer
        self.final_linear = nn.Linear(embed_dim * seq_length, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.1)

    def scaled_dot_product_attention(self, query, key, value):
        # Calculate the dot product
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply softmax to get attention scores
        attn = F.softmax(scores, dim=-1)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by value
        output = torch.matmul(attn, value)
        return output

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Perform linear operation and split into h heads

        query_l = self.linear_q(query)
        query = query_l.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot product attention
        attn_output = self.scaled_dot_product_attention(query, key, value)

        # print('1', attn_output.shape)
        # Concatenate heads and put through final linear layer
        # attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim * self.seq_length)

        # print('2', attn_output.shape)
        # combine and redistribute with a final layer
        output = self.final_linear(attn_output)

        return output, attn_output
class _TimesNet(ModelCore):
    def __init__(
            self,
            n_classes,
            n_layers,
            n_steps,
            n_features,
            top_k,
            d_model,
            d_ffn,
            n_kernels,
            dropout,
            training_loss: Criterion,
            validation_metric: Criterion,

    ):
        super().__init__()

        self.n_steps = n_steps
        self.d_model = d_model
        self.n_layers = n_layers
        self.training_loss = training_loss
        from pypots.nn.modules.loss import Criterion
        if isinstance(validation_metric, Criterion):

            self.validation_metric = self.training_loss
        else:
            self.validation_metric = validation_metric

        self.enc_embedding = DataEmbedding(
            n_features,
            d_model,
            dropout=dropout,
            n_max_steps=n_steps,
        )
        self.model = BackboneTimesNet(
            n_layers,
            n_steps,
            0,  # n_pred_steps should be 0 for the imputation task
            top_k,
            d_model,
            d_ffn,
            n_kernels,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)

        # Conv1D layers with proper kernel sizes
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool1d(1)  # Global average pooling to get fixed size output
        )
        embed_dim = 64  # Embedding size for each token
        num_heads = 1  # Number of attention heads
        seq_length = 30  # Length of the sequence
        self.seq_length = seq_length
        # Create an instance of MultiheadAttention
        self.attm = CustomMultiheadAttention(embed_dim, num_heads , seq_length)

        # Define the dimensions

        # Calculate the final projection input size
        # TimesNet output: d_model * n_steps
        # CNN output: 128 (after adaptive pooling)
        # self.projection = nn.Linear(d_model * n_steps + 128, n_classes)
        self.projection = nn.Linear(d_model , n_classes)


    def forward(
            self,
            inputs: dict,
            calc_criterion: bool = False,
    ) -> dict:
        X = inputs["X"]

        # TimesNet branch
        input_X = self.enc_embedding(X)  # [B,T,C]
        enc_out = self.model(input_X)
        timesnet_output = self.act(enc_out)

        timesnet_output = self.dropout(timesnet_output)
        # print(f'timesnet_output{timesnet_output.shape}')

        # Flatten TimesNet output
        timesnet_flat = timesnet_output.reshape(-1, self.n_steps * self.d_model)

        # CNN branch
        X_perm = X.permute(0, 2, 1)  # [B, F, T] = [32, 13, 30]
        cnn_out = self.cnn(X_perm)  # [B, 128, 1]
        cnn_out = cnn_out.permute(0, 2, 1)
        cnn_out = self.dropout(cnn_out)
        # print(f'cnn_out.shape{cnn_out.shape}')
        concated_features = torch.cat([timesnet_output, cnn_out], dim=-1)
        #combined_features, _ = self.attm(timesnet_output , cnn_out , cnn_out)
        #head1 and head2 
        combined_features, _ = self.attm(timesnet_output , concated_features , concated_features)

        combined_features = combined_features.squeeze(1)

        # print("combined_features shape:", combined_features.shape)


        # cnn_flat = cnn_out.squeeze(-1)  # [B, 128]

        # Concatenate both branches
        # combined_features = torch.cat([timesnet_flat, cnn_flat], dim=-1)

        # Final projection
        logits = self.projection(combined_features)
        # print("logits shape:", logits.shape)
        classification_proba = torch.softmax(logits, dim=1)

        results = {
            "classification_proba": classification_proba,
            "logits": logits,
        }

        if calc_criterion:
            if self.training:
                results["loss"] = self.training_loss(logits, inputs["y"])
            else:
                results["metric"] = self.validation_metric(logits, inputs["y"])

        return results
