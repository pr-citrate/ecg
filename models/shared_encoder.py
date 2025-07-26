import torch.nn as nn


def build_shared_encoder(in_channels: int = 12,
                         d_model: int = 64,
                         num_blocks: int = 2) -> nn.Sequential:
    """
    Builds the shared CNN encoder for ECG signals.

    Args:
        in_channels: Number of input channels (e.g., 12 leads)
        d_model: Number of feature channels in intermediate layers

    Returns:
        nn.Sequential model that maps (in_channels, T) -> (d_model, T/4)
    """
    layers = []
    # # Repeat Conv blocks num_blocks times
    for i in range(num_blocks):
        in_ch = in_channels if i == 0 else d_model
        layers.append(nn.Conv1d(in_ch, d_model, kernel_size=15, padding=7, stride=1))
        layers.append(nn.BatchNorm1d(d_model))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)
