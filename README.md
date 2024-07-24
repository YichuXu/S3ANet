# Spatial–Spectral Self-Attention Learning Network for Defending Against Adversarial Attacks in Hyperspectral Image Classification

PyTorch implementation of S³ANet for adversarial defenses in hyperspectral image classification.

</div>

<div align="center">
<a href='https://ieeexplore.ieee.org/document/10478963'><img src='https://img.shields.io/badge/TGRS-Paper-red'></a>

</div>

## Requirements
* Python 3.7.13
* Pytorch 1.12

## Dataset
* Download the [Pavia University image](http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat) and the corresponding [annotations](http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat). Put these files into the `Data` folder.

## Usage
* Data Preparation:
  - `python GenSample.py --train_samples 300`
 
    Prepare the training and testing set. The training samples is generated by randomly selecting `300` samples from each category.
    
* Adversarial Attack with the FGSM:
  - `CUDA_VISIBLE_DEVICES=0 python  Attack_FGSM_S3ANet.py --dataID 1 --bins 1 2 3 6 --epoch 1000 --iter 10`

* Adversarial Examples Visualization:
  - `CUDA_VISIBLE_DEVICES=0 python GenAdvExample.py --model S3ANet --bins 1 2 3 6`
  
### Reference
if you find it useful for your research, please consider giving this repo a ⭐ and citing our paper! We appreciate your support！😊

```
@article{S³ANet,
  title={S³ANet: Spatial–Spectral Self-Attention Learning Network for Defending Against Adversarial Attacks in Hyperspectral Image Classification}, 
  author={Xu, Yichu and Xu, Yonghao and Jiao, Hongzan and Gao, Zhi and Zhang, Lefei},
  journal={IEEE Trans. Geos. Remote Sens.},  
  volume={62},
  pages={1--13},
  year={2024},
}
```

## Acknowledgments
[SACNet](https://github.com/YonghaoXu/SACNet) &ensp; [FullyContNet](https://github.com/DotWang/FullyContNet) &ensp; [CCNet](https://github.com/speedinghzl/CCNet) &ensp;
