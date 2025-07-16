import torch
import torch.nn as nn


class ConceptBottleneck(nn.Module):
    """
    Concept Bottleneck layer combining attention-pooled features and prototype similarities
    to predict clinical concept scores.
    """
    def __init__(
        self,
        d_model: int,
        num_prototypes: int,
        num_concepts: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = d_model + num_prototypes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_concepts)

    def forward(self, pooled_feat: torch.Tensor, proto_sim: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_feat: Tensor of shape (batch_size, d_model)
            proto_sim: Tensor of shape (batch_size, num_prototypes)
        Returns:
            concept_scores: Tensor of shape (batch_size, num_concepts)
        """
        x = torch.cat([pooled_feat, proto_sim], dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        concept_scores = self.fc2(x)
        return concept_scores


class ClassificationHead(nn.Module):
    """
    Classification head mapping concept scores to multi-label predictions.
    """
    def __init__(
        self,
        num_concepts: int,
        num_labels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(num_concepts, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, concept_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            concept_scores: Tensor of shape (batch_size, num_concepts)
        Returns:
            out: Tensor of shape (batch_size, num_labels) with sigmoid activation
        """
        x = self.dropout(self.relu(self.fc1(concept_scores)))
        logits = self.fc2(x)
        out = self.sigmoid(logits)
        return out
