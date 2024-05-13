import itertools

import numpy as np
import torch
import torchaudio
from wv_mos_metric.metric_nets import Wav2Vec2MOS
from tqdm import tqdm


class MOSNet():
    name = "MOSNet"

    def __init__(self, sr=22050, num_splits=5, **kwargs):
        super().__init__(**kwargs)
        self.num_splits = num_splits

        self.mos_net = Wav2Vec2MOS("weights/wave2vec2mos.pth")
        self.sr = sr
        self.result = dict()

    def _compute_per_split(self, split):
        return self.mos_net.calculate(split)

    def _compute(self, samples):
        required_sr = self.mos_net.sample_rate
        resample = torchaudio.transforms.Resample(
            orig_freq=self.sr, new_freq=required_sr
        )  # TODO

        samples /= samples.abs().max(-1, keepdim=True)[0]
        samples = [resample(s).squeeze() for s in samples]

        splits = [
            samples[i: i + self.num_splits]
            for i in range(0, len(samples), self.num_splits)
        ]
        fid_per_splits = [self._compute_per_split(split) for split in splits]
        self.result["mean"] = np.mean(fid_per_splits)
        self.result["std"] = np.std(fid_per_splits)


def calculate_all_metrics(reference_wavs, metrics, n_max_files=None):
    scores = {metric.name: [] for metric in metrics}
    for x in tqdm(itertools.islice(reference_wavs, n_max_files)):
        x = torch.from_numpy(x)[None, None]
        for metric in metrics:
            metric._compute(x)
            scores[metric.name] += [metric.result["mean"]]
    scores = {k: (np.mean(v), np.std(v)) for k, v in scores.items()}
    return scores
