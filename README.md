# Mobile-like HIFI-GAN

### Антон Нуждин, курсовая работа, 3 курс

В этом репозитории находится код моей курсовой работы по улучшению HIFI-GAN. Мы представили модель, которая является эффективной модификацией оригинальной работы из [статьи](https://arxiv.org/abs/2010.05646), которая также предоставляет лучшее качество.

## Pre-requisites
1. Python >= 3.6
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Training
```
!python train.py --config config_4m.json --checkpoint_path --num_disc <amount of discriminators> --factor <factor to divide disc's channels amount>
```

Чтобы обучить другие версии модели, нужно указать другой config.json из репозитория


## Inference from wav file

```
python inference.py --checkpoint_file [generator checkpoint file path]
```
Сгенерированные аудио будут сохранены в `generated_files` 
Эту директорию можно менять с помощью опции `--output_dir`.
