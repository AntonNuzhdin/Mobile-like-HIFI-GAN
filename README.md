# Mobile-like HIFI-GAN

### Anton Nuzhdin, Coursework, 3 year

This repository contains the code for my course project on improving HIFI-GAN. We introduced a model that is an efficient modification of the original work from this [paper](https://arxiv.org/abs/2010.05646), which also delivers better quality.

![b01165ba-031f-4c1f-b072-5281301e2e72](https://github.com/user-attachments/assets/4de87f97-5c31-4a9a-bda8-b60abf6e8fb8)


## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
python train.py --config config_4m.json --checkpoint_path --num_disc <amount of discriminators> --factor <factor to divide disc's channels amount>
```

To train different model versions, just provide the corresponding config.json from the repo

## Inference from wav file

```
python inference.py --checkpoint_file [generator checkpoint file path]
```
Generated audios will be saved in `generated_files` 
One can provide custom save directiry through `--output_dir`.
