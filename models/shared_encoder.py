import torch.nn as nn


def build_shared_encoder(in_channels: int = 12, d_model: int = 64) -> nn.Sequential:
    """
    Builds the shared CNN encoder for ECG signals.

    Args:
        in_channels: Number of input channels (e.g., 12 leads)
        d_model: Number of feature channels in intermediate layers

    Returns:
        nn.Sequential model that maps (in_channels, T) -> (d_model, T/4)
    """
    layers = []
    # First Conv block
    layers.append(nn.Conv1d(in_channels, d_model, kernel_size=15, padding=7, stride=1))
    layers.append(nn.BatchNorm1d(d_model))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    # Second Conv block
    layers.append(nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, stride=1))
    layers.append(nn.BatchNorm1d(d_model))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)
