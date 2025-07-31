import os
import numpy as np
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset
import wfdb
from scipy.signal import butter, filtfilt, resample


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the signal.

    Args:
        signal: np.ndarray, shape (channels, samples)
        lowcut: float, low cutoff frequency in Hz
        highcut: float, high cutoff frequency in Hz
        fs: float, sampling frequency of the signal in Hz
        order: int, filter order

    Returns:
        Filtered signal of same shape
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal, axis=1)
    return filtered


def resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample signal from orig_fs to target_fs.

    Args:
        signal: np.ndarray, shape (..., samples)
        orig_fs: float, original sampling rate
        target_fs: float, target sampling rate

    Returns:
        Resampled signal
    """
    num_samples = int(signal.shape[-1] * target_fs / orig_fs)
    return resample(signal, num_samples, axis=-1)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per channel.

    Args:
        signal: np.ndarray, shape (channels, samples)

    Returns:
        Normalized signal
    """
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-6
    return (signal - mean) / std


class ECGDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL ECG records.

    Expects a metadata CSV with columns:
      - filename_hr: path to original 500Hz WFDB record (no extension)
      - filename_lr: path to downsampled 100Hz WFDB record (no extension)
      - labels: column containing list of binary labels or string representation

    Args:
        metadata_csv: str, path to CSV file
        data_dir: str, root directory containing WFDB files
        use_lowres: bool, if True use 100Hz data, else 500Hz
        lowcut: float, low cutoff frequency for bandpass
        highcut: float, high cutoff frequency for bandpass
        orig_fs: float, original sampling rate (500 or 100)
        target_fs: float, desired sampling rate (if different from orig_fs)
        transforms: optional callable, further transforms on signal
    """
    def __init__(
        self,
        metadata_csv: str,
        data_dir: str,
        use_lowres: bool = False,
        lowcut: float = 0.5,
        highcut: float = 40.0,
        orig_fs: float = 500.0,
        target_fs: float = 100.0,
        transforms=None,
    ):
        self.data_dir = data_dir
        self.meta = pd.read_csv(metadata_csv)
        self.use_lowres = use_lowres
        self.lowcut = lowcut
        self.highcut = highcut
        self.orig_fs = orig_fs if not use_lowres else target_fs
        self.target_fs = target_fs
        self.transforms = transforms

        # prepare file paths
        key = 'filename_lr' if use_lowres else 'filename_hr'
        self.paths = self.meta[key].tolist()
        # parse labels: if string, literal_eval; else assume list
        self.labels = self.meta['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rec_name = self.paths[idx]
        # read wfdb record (no extension)
        sig, meta = wfdb.rdsamp(os.path.join(self.data_dir, rec_name))
        signal = sig.T  # shape (channels, samples)

        # bandpass filter
        signal = bandpass_filter(signal, self.lowcut, self.highcut, self.orig_fs)

        # resample if needed
        if self.orig_fs != self.target_fs:
            signal = resample_signal(signal, self.orig_fs, self.target_fs)

        # normalize
        signal = normalize_signal(signal)

        # optional transforms
        if self.transforms:
            signal = self.transforms(signal)

        # to tensor
        signal = torch.from_numpy(signal).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return signal, label

class ECGContrastiveDataset(ECGDataset):
    def __getitem__(self, idx):
        signal, label = super().__getitem__(idx)
        view1 = self._augment(signal)
        view2 = self._augment(signal)
        return view1, view2, label

    def _augment(self, x):
        return x + 0.01 * torch.randn_like(x)
