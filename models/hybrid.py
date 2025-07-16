import torch
import torch.nn as nn

from models.shared_encoder import build_shared_encoder
from models.attention import TemporalAttention
from models.prototype import PrototypeMemory
from models.classifier import ConceptBottleneck, ClassificationHead


class HybridECGModel(nn.Module):
    """
    Hybrid ECG classification model with shared CNN encoder,
    temporal self-attention, prototype memory, concept bottleneck,
    and classification head.
    """
    def __init__(
        self,
        in_channels: int = 12,
        d_model: int = 64,
        num_prototypes: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        num_concepts: int = 10,
        num_labels: int = 5,
        concept_hidden: int = 128,
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        # Shared feature encoder
        self.encoder = build_shared_encoder(in_channels, d_model)
        # Temporal self-attention branch
        self.attention = TemporalAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_len=max_len,
        )
        # Prototype memory branch
        self.prototype = PrototypeMemory(d_model, num_prototypes)
        # Concept bottleneck layer
        self.bottleneck = ConceptBottleneck(
            d_model=d_model,
            num_prototypes=num_prototypes,
            num_concepts=num_concepts,
            hidden_dim=concept_hidden,
            dropout=dropout,
        )
        # Classification head
        self.classifier = ClassificationHead(
            num_concepts=num_concepts,
            num_labels=num_labels,
            hidden_dim=classifier_hidden,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass for HybridECGModel.

        Args:
            x: Tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            dict containing:
                'logits': multi-label predictions (batch_size, num_labels)
                'concept_scores': concept layer outputs (batch_size, num_concepts)
                'proto_sim': prototype similarities (batch_size, num_prototypes)
                'attn_feats': attention feature map (batch_size, d_model, seq_len/4)
        """
        # Shared encoding
        feats = self.encoder(x)  # (batch, d_model, seq_len/4)
        # Attention branch
        attn_feats = self.attention(feats)  # (batch, d_model, seq_len/4)
        # Pool attention features
        pooled = attn_feats.mean(dim=2)  # (batch, d_model)
        # Prototype branch (uses raw feats)
        proto_sim = self.prototype(feats)  # (batch, num_prototypes)
        # Concept bottleneck
        concept_scores = self.bottleneck(pooled, proto_sim)  # (batch, num_concepts)
        # Classification head
        logits = self.classifier(concept_scores)  # (batch, num_labels)

        return {
            'logits': logits,
            'concept_scores': concept_scores,
            'proto_sim': proto_sim,
            'attn_feats': attn_feats,
        }
