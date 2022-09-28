# MetaDetect3D
MetaDetect3D is a post-processing tool for 3D object detection neural networks that improve calibration and performance.

## Preface

<p>Advances in the development of deep neural networks (NNs) over the last two decades have made it possible for an artificial intelligence (AI) not only to tackle increasingly complex tasks, but even to master them at a human level. Computer vision, natural language processing and reinforcement learning approaches are just some of the few areas covered by the now huge field of AI. Above all, the task of autonomous driving keeps scientists around the world engaged in perfecting its capabilities and reducing its weaknesses, with the goal of making it safer than any human driver. While the two main approaches, visual and LiDAR (light detection and ranging) based, already work well in many traffic scenarios, there are still many cases where the deep learning model makes incorrect predictions. In particular, it often predicts an object where none is (false positive), or it predicts no object where one should have been (false negative). Both of these errors can be very dangerous in practice.</p>
<p>This repository presents the work of my Master Thesis about Meta Detection methods. MetaDetect3D is inspired by <a href="https://github.com/schubertm/MetaDetect">MetaDetect</a> created and developed at the University of Wuppertal in Germany.</p>
<p>MetaDetect3D is a post-processing tool for 3D object detection neural networks. The idea behind MetaDetect3D is to use informations provided by an 3D Object Detector, specifically its input, output and from its proposals in order to define features and metrics on those in order to create a new dataset. The new dataset is used to train a new ML model which further refines the previous output of the underlying object detector.</p>
<p>Meta models can have two purposes: The first is to evaluate the predictions by assigning them either a numerical value between 0 and 1 (Meta regression) or a binary label or probability (Meta classification). This functionality is crucial if one does not want to rely solely on the confidence score output by the object detector. If trained properly, this can lead to a better quality estimate of the predictions obtained from the object detector. Second, one can use the Meta detector to spot misplaced or missing labels, or stated differently, to detect label errors. This can be done by assuming that the Meta regressor assigns high values, or equivalently that the Meta classifier assigns a true label to boxes where it assumes the prediction is correct, even though the ground truth does not contain the label. This would yield potentially missing labels. Similarly, the Meta models should assign low values (or a false label) to a prediction where no box is assumed even though a box was predicted. In this case, if the prediction overlaps with a ground truth box, the label may be incorrect.</p>
<p>The graph below displays the workflow of MetaDetect3D in detail.</p>

<p align="center">
 <img src="https://github.com/JanMarcelKezmann/MetaDetect3D/blob/main/images/MetaDetect3D%20Pipeline.png" width="717" height="679">
</p>

<p>In words the steps of the MetaDetect3D methods are:<p>

1) Train a object detector on training set (and validate it on validation set)
2) Crawling prediction 3D object detect and labels of validation and test set
3) Computation of Meta dispersion measures on previously crawled data
4) MetaDetect3D training pipeline including parameter search (on validation set)
5) MetaDetect3D evaluation on best Meta models (on test set)

### Main Library Features

- High Level API
- Meta Regression and Classification possible
- Simple Post-Processing Tool for any 3D Object Detector
- Various Machine Learning Models available

## Table of Contents

 - [Installation and Setup](#installation-and-setup)
 - [Usage and Examples](#usage-and-examples)
 - [Citing](#citing)
 - [License](#license)
 - [References](#references)
 - [Authors and Acknowledgements](#authors-and-acknowledgements)
 - [Support](#support)

## Installation and Setup

<p>To get the repository running just check the following requirements.</p>

**Requirements**
**Windows or Linus**
1) Python 3.6 or higher
2) torch >= 1.10.1
3) scikit-learn >= 0.24.1
3) numpy >= 1.19.5
4) pandas >= 1.15
5) mmcv-full >= 1.4.6
6) mmcls >= 0.21.0
7) mmdet >= 2.22.0
8) mmdet3d >= 1.0.0rc0
9) mmsegmentation >= 0.22.1

<p>Furthermore just execute the following command to download and install the git repository. </p>
<p>This will not install any open-mmlab package like mmcv, mmdet3d, etc.. This has to be done manually.</p>

**Clone Repository**

    $ git clone https://github.com/JanMarcelKezmann/MetaDetect3D.git

or directly install it:<br>
**Pip Install Repository**

    $ pip install git+https://github.com/JanMarcelKezmann/MetaDetect3D.git

## Usage and Examples

Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

- [Jupyter Notebbok] MetaDetect3D on KITTI Dataset <a href= "https://github.com/JanMarcelKezmann/MetaDetect3D/blob/master/examples/MetaDetect3D_KITTI_Example.ipynb">here</a>

## Available ML Models


CLASSIFICATION_MODELS = ['logistic', 'ridge_cls', 'random_forest_cls', 'gb_cls', "mlp_cls"]
REGRESSION_MODELS = ['ridge_reg', 'lasso_reg', 'random_forest_reg', 'gb_reg', "mlp_reg"]

|         Model         |     Classifier     |     Regressors     |
|-----------------------|--------------------|--------------------|
|**Logistic**           | :heavy_check_mark: |                    |
|**Ridge**              | :heavy_check_mark: | :heavy_check_mark: |
|**Random Forest**      | :heavy_check_mark: | :heavy_check_mark: |
|**Gradient Boosting**  | :heavy_check_mark: | :heavy_check_mark: |
|**MLP**                | :heavy_check_mark: | :heavy_check_mark: |


## Object Detectors

 - PointPillars
 - CenterPoint

## Predefined Dataset Pipelines

 - KITTI
 - NuScenes
 - (APTIV) custom dataset
 
 This list should help you for adapting the code on custom datasets on your own.

## Citing

    @misc{Kezmann:2022,
      Author = {Jan-Marcel Kezmann},
      Title = {MetaDetect3D},
      Year = {2022},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/JanMarcelKezmann/MetaDetect3D}}
    } 
    
## License

Project is distributed under <a href="https://github.com/JanMarcelKezmann/MetaDetect3D/blob/master/LICENSE">MIT License</a>.

## References

<p>Thank you for all the papers and people that made this repository possible.</p>

## Authors and Acknowledgment

## Support

Feel free to raise an Issue in case something does not work properly.

