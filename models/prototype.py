import torch
import torch.nn as nn

class PrototypeMemory(nn.Module):
    """
    Prototype memory module for case-based explanations.

    Stores learnable prototype vectors and computes similarity scores
    between input features and prototypes.
    """
    def __init__(self, d_model: int, num_prototypes: int):
        """
        Args:
            d_model: Dimension of the feature vector
            num_prototypes: Number of prototype vectors to learn
        """
        super().__init__()
        self.d_model = d_model
        self.num_prototypes = num_prototypes
        # Initialize prototype vectors
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity scores between features and prototypes.

        Args:
            features: Tensor of shape (batch_size, d_model) or (batch_size, d_model, L)

        Returns:
            sim: Tensor of shape (batch_size, num_prototypes), where
                 sim[b, j] = -||features[b] - prototypes[j]||_2
        """
        # If temporal features provided, average pool over time dimension
        if features.dim() == 3:
            # features: (batch, d_model, L)
            features = features.mean(dim=2)  # -> (batch, d_model)

        # features: (batch, d_model), prototypes: (num_prototypes, d_model)
        # Compute pairwise distances
        # Expand dims for broadcasting
        # (batch, 1, d_model) - (1, num_prototypes, d_model)
        diff = features.unsqueeze(1) - self.prototypes.unsqueeze(0)
        # Euclidean distance
        distances = diff.norm(dim=2)  # (batch, num_prototypes)
        # Similarity as negative distance
        sim = -distances
        return sim
