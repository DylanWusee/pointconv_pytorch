# PointConv
**PointConv: Deep Convolutional Networks on 3D Point Clouds.** CVPR 2019  
Wenxuan Wu, Zhongang Qi, Li Fuxin.

## Introduction
This project is based on our CVPR2019 paper. You can find the [arXiv](https://arxiv.org/abs/1811.07246) version here.

```
@inproceedings{wu2019pointconv,
  title={Pointconv: Deep convolutional networks on 3d point clouds},
  author={Wu, Wenxuan and Qi, Zhongang and Fuxin, Li},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9621--9630},
  year={2019}
}
```

Unlike images which are represented in regular dense grids, 3D point clouds are irregular and unordered, hence applying convolution on them can be difficult. In this paper, we extend the dynamic filter to a new convolution operation, named PointConv. PointConv can be applied on point clouds to build deep convolutional networks. We treat convolution kernels as nonlinear functions of the local coordinates of 3D points comprised of weight and density functions. With respect to a given point, the weight functions are learned with multi-layer perceptron networks and the density functions through kernel density estimation. A novel reformulation is proposed for efficiently computing the weight functions, which allowed us to dramatically scale up the network and significantly improve its performance. The learned convolution kernel can be used to compute translation-invariant and permutation-invariant convolution on any point set in the 3D space. Besides, PointConv can also be used as deconvolution operators to propagate features from a subsampled point cloud back to its original resolution. Experiments on ModelNet40, ShapeNet, and ScanNet show that deep convolutional neural networks built on PointConv are able to achieve state-of-the-art on challenging semantic segmentation benchmarks on 3D point clouds. Besides, our experiments converting CIFAR-10 into a point cloud showed that networks built on PointConv can match the performance of convolutional networks in 2D images of a similar structure.

## Installation
The code is modified from repo [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Please install [PyTorch](https://pytorch.org/), [pandas](https://pandas.pydata.org/), and [sklearn](https://scikit-learn.org/).
The code has been tested with Python 3.5, pytorch 1.2, CUDA 10.0 and cuDNN 7.6 on Ubuntu 16.04.

## Usage
### ModelNet40 Classification

Download the ModelNet40 dataset from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip). This dataset is the same one used in [PointNet](https://arxiv.org/abs/1612.00593), thanks to [Charles Qi](https://github.com/charlesq34/pointnet). Copy the unziped dataset to ```./data/modelnet40_normal_resampled```.

To train the model,
```
python train_cls_conv.py --model pointconv_modelnet40 --normal
```

To evaluate the model,
```
python eval_cls_conv.py --checkpoint ./checkpoints/checkpoint.pth --normal
```

## License
This repository is released under MIT License (see LICENSE file for details).



