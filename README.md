# Face-Driven Zero-Shot Voice Conversion with Memory-based Face-Voice Alignment

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2309.09470)
[![githubio](https://img.shields.io/static/v1?message=Audio%20Samples&logo=Github&labelColor=grey&color=blue&logoColor=white&label=%20&style=flat)](https://levent9.github.io/ZeroshotFaceVC-demo/)

This [paper](https://arxiv.org/pdf/2309.09470) presents a novel task, zero-shot voice conversion based on face images (zero-shot FaceVC). We leverage a memory-based face-voice alignment module for the capture of voice characteristics from face images.  A mixed supervision strategy is also introduced to mitigate the long-standing issue of the inconsistency between training and inference phases for voice conversion tasks. To obtain speaker-independent content-related representations, we transfer the knowledge from a pretrained zero-shot voice conversion model [VQMIVC](https://github.com/Wendison/VQMIVC) to our zero-shot FaceVC model. 

[Paper Demo](https://levent9.github.io/ZeroshotFaceVC-demo/)


## Training
- Step1. Data preparation & preprocessing
1. Put LRS3 corpus under directory "Dataset/LRS3"
2. Extract wav from LRS3 video
```python
python Tools/preprocess/extract_wav_from_video.py 
```
3. Extract mel and lf0 from wav
```python
python Tools/preprocess/extract_wav_feature.py
```
4. Extract face feature 
```python
python Tools/Preprocess/extract_face_feature.py
```
5. Extract speech feature
```
python Tools/Preprocess/extract_spk_emb.py
```

- Step2. Model training
1. ParallelWaveGAN is used as the vocoder, so firstly please install [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

2. Download the pretrained [VQMIVC](https://drive.google.com/drive/folders/1u8xAJdJEQ3MKfTDSks1xFkTcR2CXdfAd?usp=sharing) and place it in folder pretrained

3. Training model
```
./run_shell/train.sh
```
- Step3. Inference 
1. Preprocess the samples for inference following Step 1. The IDs of the preprocessed samples can be found in the files "test_src_speakers.txt" and "test_tar_speakers.txt."

2. Runing inference
```
./run_shell/inference.sh
```



## Citation
If the code is used in your research, please <a class="github-button" href="https://github.com/wendison/VQMIVC" data-icon="octicon-star" aria-label="Star wendison/VQMIVC on GitHub">Star</a> our repo and cite our paper:
```
@inproceedings{10.1145/3581783.3613825,
author = {Sheng, Zheng-Yan and Ai, Yang and Chen, Yan-Nian and Ling, Zhen-Hua},
title = {Face-Driven Zero-Shot Voice Conversion with Memory-Based Face-Voice Alignment},
year = {2023},
isbn = {9798400701085},
url = {https://doi.org/10.1145/3581783.3613825},
doi = {10.1145/3581783.3613825},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {8443–8452},
location = {Ottawa ON, Canada},
}
```

## Acknowledgements:
* The voice conversion backbone is borrowed from [VQMIVC](https://github.com/Wendison/VQMIVC)
* The vocoder is borrowed from [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)