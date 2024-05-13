import torch
import os
import argparse
from tqdm import tqdm
import torchaudio

from wv_mos_metric.utils import closest_power_of_two, load_wav, get_device
from wv_mos_metric.metric import MOSNet
from wv_mos_metric.metric import calculate_all_metrics


def compute_metrics():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_wavs_dir")
    a = parser.parse_args()
    fake_wavs_dir = a.fake_wavs_dir
    metrics = [MOSNet()]

    device = get_device()
    filelist = os.listdir(fake_wavs_dir)
    fake_samples = []
    for i, filename in tqdm(enumerate(filelist)):
        if i == 10:
            break
        wav, _ = torchaudio.load(os.path.join(fake_wavs_dir, filename))
        wav = wav.to(device)

        pad_size = closest_power_of_two(wav.shape[-1]) - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, pad_size))
        fake_samples.append(wav.squeeze().cpu().numpy())

    scores = calculate_all_metrics(fake_samples, metrics)
    print(scores)
    return fake_samples


if __name__ == '__main__':
    compute_metrics()
