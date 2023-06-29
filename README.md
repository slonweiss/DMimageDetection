# On the detection of synthetic images generated by diffusion models

[![Github](https://img.shields.io/badge/Github%20webpage-222222.svg?style=for-the-badge&logo=github)](https://grip-unina.github.io/DMimageDetection/)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2211.00680)
[![IEEE](https://img.shields.io/badge/-IEEE-6093BF.svg?style=for-the-badge)](https://doi.org/10.1109/ICASSP49357.2023.10095167)
[![GRIP](https://img.shields.io/badge/-GRIP-0888ef.svg?style=for-the-badge)](https://www.grip.unina.it)

<p align="center">
 <img src="./docs/preview.png" alt="preview" width="500pt" />
</p>

This is the official repository of the paper:
[On the detection of synthetic images generated by diffusion models](https://arxiv.org/abs/2211.00680) 
Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, Luisa Verdoliva

## Test-set
The synthetic images used as test can be downloaded from the following [link](https://drive.google.com/file/d/1grvgKiIq0ny8ImQzSUXPk3nd-AMEDjNb/view?usp=share_link) alongside a csv file stating the processing applied in the paper on each image. The real images can be downloaded from the following freely available datasets : [IMAGENET](https://image-net.org/index.php), [UCID](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/5307/0000/UCID-an-uncompressed-color-image-database/10.1117/12.525375.short),[COCO - Common Objects in Context](https://cocodataset.org/#home).
The real images should then be placed in a folder with the same name that has been recorded in the csv file
The directory containing the test set should have the following structure:
```
Testset directory
|--biggan_256
|--biggan_512
.
.
.
|--real_coco_valid
|--real_imagenet_valid
|--real_ucid
.
.
.
|--taming-transformers_segm2image_valid
```

The annotations used to generate images with text to images models belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License (https://cocodataset.org/#termsofuse).

## Traning-set

#### 1) ProGAN Traning-set
For training using ProGAN images, we used the traning-set provided by ["CNN-generated images are surprisingly easy to spot...for now"](https://github.com/PeterWang512/CNNDetection)
 
#### 2) Latent Diffusion Traning-set
For training using Latent Diffusion images, we generated 200K fake images, while the 200K real images come from two public datasets, [COCO - Common Objects in Context](https://cocodataset.org/#home) and  [LSUN - Large-scale Scene Understanding](https://www.yf.io/p/lsun).

## Code
In this repository it is also provided a python script to apply on each image the processing outlined by the csv file.

There are also provided the code to test the networks on the provided images.  
The networks weights can be downloaded from the following [link](https://drive.google.com/file/d/1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf/view?usp=share_link) 

In order to launch the scripts, create a conda enviroment using the enviroment.yml provided.

The commands can be launched as follows:

To generate the images modified according to the details contained in the csv file, launch the script as follows:

```
python csv_operations.py --data_dir /path/to/testset/dir --out_dir /path/to/output/dir --csv_file /path/to/csv/file
```
In order to calculate the outputs of each model launch the script as shown below
```
python main.py --data_dir /path/to/testset/dir --out_dir /path/to/output/dir --csv_file /path/to/csv/file

```
The output CSV contains the logit values provided by the networks. The image is detected fake if the logit value is positive.
Finally to generate the csv files containing the accuracies and aucs calculated per detection method and per generator architecture launche the last script as described.
```
python metrics_evaluations.py --data_dir /path/to/testset/dir --out_dir /path/to/output/dir
```

## Requirements

![Pytorch](https://img.shields.io/badge/Pytorch-grey.svg?style=plastic)
![Matplotlib](https://img.shields.io/badge/Matplotlib-grey.svg?style=plastic)
![tqdm](https://img.shields.io/badge/tqdm-grey.svg?style=plastic)
![Pillow](https://img.shields.io/badge/Pillow-grey.svg?style=plastic)
![numpy](https://img.shields.io/badge/numpy-grey.svg?style=plastic)

## Overview

Over the past decade, there has been tremendous progress in creating synthetic media, mainly thanks to the development of powerful methods based on generative adversarial networks (GAN). Very recently, methods based on diffusion models (DM) have been gaining the spotlight. In addition to providing an impressive level of photorealism, they enable the creation of text-based visual content, opening up new and exciting opportunities in many different application fields, from arts to video games. On the other hand, this property is an additional asset in the hands of malicious users, who can generate and distribute fake media perfectly adapted to their attacks, posing new challenges to the media forensic community. With this work, we seek to understand how difficult it is to distinguish synthetic images generated by diffusion models from pristine ones and whether current state-of-the-art detectors are suitable for the task. To this end, first we expose the forensics traces left by diffusion models, then study how current detectors, developed for GAN-generated images, perform on these new synthetic images, especially in challenging social-network scenarios involving image compression and resizing.

## License

The license of the code can be found in the LICENSE.md file.

The annotations used to generate images with text to images models belong to the COCO Consortium and are licensed under a Creative Commons Attribution 4.0 License (https://cocodataset.org/#termsofuse).

## Bibtex 

```
@InProceedings{Corvi_2023_ICASSP,
  author={Corvi, Riccardo and Cozzolino, Davide and Zingarini, Giada and Poggi, Giovanni and Nagano, Koki and Verdoliva, Luisa},
  title={On The Detection of Synthetic Images Generated by Diffusion Models},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095167}
}
```



