# import torch
# from torch import nn
# from typing import Optional

# def Attention(n_in: int, n_out: int, n_latent: Optional[int] = None) -> nn.Module:
#     if n_latent == 0:
#         return nn.Linear(n_in, n_out)  # Output 4 attention scores per instance
#     else:
#         n_latent = n_latent or (n_in + 1) // 2
#         return nn.Sequential(
#             nn.Linear(n_in, n_latent),
#             nn.Tanh(),
#             nn.Linear(n_latent, n_out)  # 4 scores for 4 classes
#         )

# class Attention_MIL(nn.Module):
#     def __init__(
#         self,
#         n_feats: int,
#         n_out: int,
#         z_dim: int = 256,
#         dropout_p: float = 0.5,
#         encoder: Optional[nn.Module] = None,
#         attention: Optional[nn.Module] = None,
#         head: Optional[nn.Module] = None,
#         attention_gate: float = 0,
#         temperature: float = 1.
#     ) -> None:
#         super().__init__()
#         self.n_out=n_out
#         self.encoder = encoder or nn.Sequential(
#             nn.Linear(n_feats, z_dim),
#             nn.ReLU()
#         )
#         self.attention = attention or Attention(z_dim, n_out)
#         self.head = head or nn.Sequential(
#             nn.Flatten(start_dim=1, end_dim=-1),
#             nn.BatchNorm1d(z_dim * n_out),  # Handle concatenated class features
#             nn.Dropout(dropout_p),
#             nn.Linear(z_dim * n_out, n_out)
#         )
#         self._neg_inf = torch.tensor(-torch.inf)
#         self.attention_gate = attention_gate
#         self.temperature = temperature

#     def forward(self, bags, lens, *, return_attention=False):
#         assert bags.ndim == 3
#         assert bags.shape[0] == lens.shape[0]

#         # Encode instance features
#         embeddings = self.encoder(bags)  # (BS, BAG_SIZE, Z_DIM)
#         batch_size, bag_size, z_dim = embeddings.shape
        
#         # Calculate attention scores for each class
#         raw_attention = self.attention(embeddings)  # (BS, BAG_SIZE, 4)
        
#         # Apply instance mask with negative infinity
#         attention_mask = self._create_attention_mask(lens, bag_size, embeddings.device)
#         masked_attention = torch.where(
#             attention_mask,
#             raw_attention,
#             torch.full_like(raw_attention, self._neg_inf)
#         )

#         # Apply attention gating if specified
#         if self.attention_gate and bag_size > 1:
#             thresholds = torch.quantile(
#                 masked_attention,
#                 q=self.attention_gate,
#                 dim=1,
#                 keepdim=True
#             )
#             masked_attention = torch.where(
#                 masked_attention > thresholds,
#                 masked_attention,
#                 torch.full_like(masked_attention, self._neg_inf)
#             )

#         # Apply temperature and softmax to get attention weights
#         attention_weights = torch.softmax(masked_attention / self.temperature, dim=1)  # (BS, BAG_SIZE, 4)

#         # Aggregate features per class using attention weights
#         class_features = []
#         for cls_idx in range(self.n_out):
#             cls_weights = attention_weights[..., cls_idx].unsqueeze(-1)  # (BS, BAG_SIZE, 1)
#             weighted_features = cls_weights * embeddings  # (BS, BAG_SIZE, Z_DIM)
#             class_features.append(weighted_features.sum(dim=1))  # (BS, Z_DIM)
        
#         # Concatenate class-specific features and make predictions
#         final_features = torch.cat(class_features, dim=1)  # (BS, Z_DIM*4)
#         scores = self.head(final_features)  # (BS, N_OUT)

#         if return_attention:
#             return scores, attention_weights
#         return scores

#     def _create_attention_mask(self, lens, bag_size, device):
#         batch_size = lens.shape[0]
#         index_matrix = torch.arange(bag_size, device=device).expand(batch_size, -1)
#         return (index_matrix < lens.unsqueeze(-1)).unsqueeze(-1)  # (BS, BAG_SIZE, 1)






import torch
from torch import nn
from typing import Optional

# Set the random seed for reproducibility
torch.manual_seed(42)

def Attention(n_in: int, n_out: int, n_latent: Optional[int] = None) -> nn.Module:
    if n_latent == 0:
        return nn.Linear(n_in, n_out)  # Output 4 attention scores per instance
    else:
        n_latent = n_latent or (n_in + 1) // 2
        return nn.Sequential(
            nn.Linear(n_in, n_latent),
            nn.GELU(),
            nn.Linear(n_latent, n_out)  # 4 scores for 4 classes
        )

class Attention_MIL(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_out: int,
        z_dim: int = 256,
        dropout_p: float = 0.5,
        encoder: Optional[nn.Module] = None,
        attention: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        attention_gate: float = 0,
        temperature: float = 1.0
    ) -> None:
        super().__init__()
        self.n_out = n_out
        self.encoder = encoder or nn.Sequential(
            nn.Linear(n_feats, z_dim),
            nn.ReLU()
        )
        self.attention = attention or Attention(z_dim, n_out)
        self.head = head or nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.BatchNorm1d(z_dim * n_out),  # Handle concatenated class features
            nn.Dropout(dropout_p),
            nn.Linear(z_dim * n_out, 256),
            nn.ReLU(),
            nn.Linear(256, n_out)
        )
        self._neg_inf = torch.tensor(-torch.inf)
        self.attention_gate = attention_gate
        self.temperature = temperature

    def forward(self, bags, lens, *, return_attention=False):
        assert bags.ndim == 3
        assert bags.shape[0] == lens.shape[0]

        # Encode instance features
        embeddings = self.encoder(bags)  # (BS, BAG_SIZE, Z_DIM)
        batch_size, bag_size, z_dim = embeddings.shape
        
        # Calculate attention scores for each class
        raw_attention = self.attention(embeddings)  # (BS, BAG_SIZE, 4)
        
        # Apply instance mask with negative infinity
        attention_mask = self._create_attention_mask(lens, bag_size, embeddings.device)
        masked_attention = torch.where(
            attention_mask,
            raw_attention,
            torch.full_like(raw_attention, self._neg_inf)
        )

        # Apply attention gating if specified
        if self.attention_gate and bag_size > 1:
            thresholds = torch.quantile(
                masked_attention,
                q=self.attention_gate,
                dim=1,
                keepdim=True
            )
            masked_attention = torch.where(
                masked_attention > thresholds,
                masked_attention,
                torch.full_like(masked_attention, self._neg_inf)
            )

        # Apply temperature and softmax to get attention weights
        attention_weights = torch.softmax(masked_attention / self.temperature, dim=1)  # (BS, BAG_SIZE, 4)

        # Aggregate features per class using attention weights
        class_features = []
        for cls_idx in range(self.n_out):
            cls_weights = attention_weights[..., cls_idx].unsqueeze(-1)  # (BS, BAG_SIZE, 1)
            weighted_features = cls_weights * embeddings  # (BS, BAG_SIZE, Z_DIM)
            class_features.append(weighted_features.sum(dim=1))  # (BS, Z_DIM)
        
        # Concatenate class-specific features and make predictions
        final_features = torch.cat(class_features, dim=1)  # (BS, Z_DIM*4)
        scores = self.head(final_features)  # (BS, N_OUT)

        if return_attention:
            return scores, attention_weights
        return scores

    def _create_attention_mask(self, lens, bag_size, device):
        batch_size = lens.shape[0]
        index_matrix = torch.arange(bag_size, device=device).expand(batch_size, -1)
        return (index_matrix < lens.unsqueeze(-1)).unsqueeze(-1)  # (BS, BAG_SIZE, 1)