import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader

class EnhancedClassificationHead(nn.Module):
    """
    Enhanced classification head with multiple layers and residual connections.
    """
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.fc1 = nn.Linear(512, input_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(0.3)  # Reduced dropout rate
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.batch_norm = nn.BatchNorm1d(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, num_labels)
        self.residual = nn.Linear(512, num_labels)  # Residual connection

    def forward(self, x):
        residual = self.residual(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.fc3(x)
        return x + residual  # Combine with residual connection


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.out_proj(out)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2


class ImprovedVideoClipModel(nn.Module):
    def __init__(self, clip_model, classification_head, dim=512):
        super().__init__()
        self.clip_model = clip_model
        self.classification_head = classification_head
        
        self.temporal_pos_enc = PositionalEncoding(dim)
        self.frame_attention = MultiHeadCrossAttention(dim)
        self.cross_attention = MultiHeadCrossAttention(dim)
        self.gated_fusion = GatedFusion(dim)
        
        self.feature_pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            ) for _ in range(3)
        ])
        
    def forward(self, features, captions):
        # Get text features
        text_features = self.clip_model.get_text_features(captions)
        
        # Add positional encoding to video features
        features = self.temporal_pos_enc(features)
        
        # Multi-scale feature processing
        pyramid_features = []
        for layer in self.feature_pyramid:
            features = layer(features)
            pyramid_features.append(features)
        
        # Frame-level self attention
        frame_features = self.frame_attention(features, features, features)
        
        # Cross attention with text
        text_features = text_features.unsqueeze(1).repeat(1, frame_features.size(1), 1)
        cross_attn_features = self.cross_attention(frame_features, text_features, text_features)
        
        # Combine features from different scales
        combined_pyramid = sum(pyramid_features)
        
        # Gated fusion of attention features and pyramid features
        final_features = self.gated_fusion(cross_attn_features, combined_pyramid)
        
        # Pool temporal dimension
        final_features = final_features.mean(dim=1)
        
        return self.classification_head(final_features)
