import torch
from scipy.io.wavfile import read


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate
