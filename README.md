***

# Multi-label Deep Learning Assignment

### Problem Statement: Predicting Deep Fashion Attributes

(see problem description in the given .pdf)

**Important Considerations and Planned Approach**

- Each garment has 3 attributes (*neck type, sleeve length, and pattern*).
- Each attribute has their own classes (*neck attribute has 0-6 classes, sleeve length attribute has 0-3 classes, and pattern attribute has 0-9 classes*).
- Thus, the total number of classes is 7 + 4 + 10 = 21.
- This is a **multi-label** problem, and the neural network should output the 3 predicted attributes, which consist of a total of 21 classes, given the image.
- Only a **single neural network** should be used (*so the neural network should learn to extract the features of all the 3 attributes by itself, eg. only 1 backbone is to be used*). 
- There are **plenty of missing/unknown labels** which are labelled as '#N/A' (*see the given .csv*).
- To achieve best usage of existing data with minimum bias, data which has '#N/A' as its attribute should not be discarded.
- Upon checking the distribution of the dataset (*see 'Usage' instruction below*), it can be seen that there are **high class imbalance** (*eg. no. neck 6, sleeve length 3, and pattern 9 are much higher than their attribute's counterpart*).
- In addition, **uncommon classes of an attribute are usually paired with common classes of another attribute** (*eg. neck 6 with sleeve length 3 with pattern 0*).
- When data augmentation is performed on data with uncommon classes of an attribute, the number of common classes of another attribute paired with it increases too.
- **Some incorrect images are manually removed** from the given .zip so that they are not further augmented.
- **To improve the class balance, a proposed data augmentation algorithm (*see 'Usage' instruction below*) is implemented.** This results in a slight improvement in the class balancing (*see './pics/cod_neck.PNG', './pics/cod_sleeve_length.PNG', and './pics/cod_pattern.PNG'*).
- Although the data augmentation helps to better balance the classes while increasing training data, **it increases the occurence of '#N/A' labels**.
- As a result, this is a **multi-label with partial labelling** problem.
- A [neural network](https://github.com/Alibaba-MIIL/PartialLabelingCSL) is selected and adapted for this problem.

### Installation

1. Ensure that the operating PC has GPU. Microsoft Visual Studio and its C++ build tools must be installed. Also, NVIDIA CUDA must be system-installed (*not just in the Anaconda environment, eg. conda install cudatoolkit=10.2 -c pytorch*). This is because some parts of InPlace-ABN (*a novel batch normalization approach which is used in the selected neural network model*) have native C++/CUDA implementations.

2. Create a conda environment with Python 3.7.x (*should be fine as long as PyTorch can be properly installed*).

```conda create -n flixstock python=3.7```

3. Install PyTorch using conda.

```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch```

4. Install other dependencies (*might have more required dependencies but just install them on the fly when the model is run later*).

```
cd PartialLabelingCSL
pip install -r requirements.txt
```

5. Move the 'Datasets' folder out of the main 'PartialLabelingCSL'. This is to ensure that the path variables in the code are correct (*I always put my datasets outside the repository so that they can be shared across other repositories without confusion.*). In summary, 

```
Flixstock Assignment/
    Datasets/
        FlixstockTask/
            ...
    PartialLabelingCSL/
        data/
            ...
        outputs/
            ...
        ...
    <other repository>/
        ...
    ...
```

6. Download the dataset and annotations [here](https://drive.google.com/drive/folders/1xJkBz5i_oBbOET0FqQq1IBeH8RePGXHc?usp=sharing) and place it into 'Datasets/FlixstockTask'. Extract the 'images_cleaned.zip' and ensure that the final directory looks like

```
Flixstock Assignment/
    Datasets/
        FlixstockTask/
            images_cleaned/
                images/
                    <cloth 1.jpg>
                    ...
            attributes_cleaned.csv
            images_train_test_valid_split.py
    PartialLabelingCSL/
        data/
            ...
        outputs/
            ...
        ...
    <other repository>/
        ...
    ...
```

7. Download the Flixstock pretrained weights [here](https://drive.google.com/drive/folders/13tLaXn8SyP-jjQIg6VKOsadYaL3wdFAi?usp=sharing) and place it in the './pretrained_weights' folder.

```
Flixstock Assignment/
    Datasets/
        FlixstockTask/
            images_cleaned/
                images/
                    <cloth 1.jpg>
                    ...
            attributes_cleaned.csv
            images_train_test_valid_split.py
    PartialLabelingCSL/
        data/
            ...
        outputs/
            ...
        pretrained_weights/
            <weight.ckpt>
            ...
        ...
    <other repository>/
        ...
    ...
```

8. Done with installation!

### Usage

**Data Preparation**

Note: This must be ran before performing training, validation, and inference. Code is commented.

1. Navigate to 'Datasets/FlixstockTask'
2. ```python images_train_test_valid_split.py```

**Training**

Note: The training was performed lightly, without parameter tuning. Default parameters from the original code was used, with the required change to use Flixstock images on the model.

1. Navigate to '../PartialLabelingCSL'
2. ```python train.py```

**Validation**

1. Navigate to '../PartialLabelingCSL'
2. ```python validate.py```

Current mAP score: 41.47331810982931 (no parameter tuning done)

**Inference**

Note: When args.all_val_inference is True, the model will infer will validation and output the prediction to './results/attributes_val_inference.csv'. If not, single image inference is also possible (*set image path in 'infer.py'*).

1. Navigate to '../PartialLabelingCSL'
2. ```python infer.py --all_val_inference```

**Future Direction**

- Perform parameter fine-tuning to optimize validation score
- Explore other novel multi-label classification approach (eg. graph networks) with consideration of methods that can assist in training heavily partial annotated dataset


***

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-label-classification-with-partial/multi-label-classification-on-openimages-v6)](https://paperswithcode.com/sota/multi-label-classification-on-openimages-v6?p=multi-label-classification-with-partial)

# Multi-label Classification with Partial Annotations using Class-aware Selective Loss

<br> [Paper](https://arxiv.org/abs/2110.10955) |
[Pretrained models](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/README.md#pretrained-models) |
[OpenImages download](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/OpenImages.md)

Official PyTorch Implementation

> Emanuel Ben-Baruch, Tal Ridnik, Itamar Friedman, Avi Ben-Cohen, Nadav Zamir, Asaf Noy, Lihi Zelnik-Manor<br/> DAMO Academy, Alibaba
> Group

**Abstract**

Large-scale multi-label classification datasets are commonly, and perhaps inevitably, partially annotated. That is, only a small subset of labels are annotated per sample.
Different methods for handling the missing labels induce different properties on the model and impact its accuracy.
In this work, we analyze the partial labeling problem, then propose a solution based on two key ideas. 
First, un-annotated labels should be treated selectively according to two probability quantities: the class distribution in the overall dataset and the specific label likelihood for a given data sample.
We propose to estimate the class distribution using a dedicated temporary model, and we show its improved efficiency over a naive estimation computed using the dataset's partial annotations.
Second, during the training of the target model, we emphasize the contribution of annotated labels over originally un-annotated labels by using a dedicated asymmetric loss.
Experiments conducted on three partially labeled datasets, OpenImages, LVIS, and simulated-COCO, demonstrate the effectiveness of our approach. Specifically, with our novel selective approach, we achieve state-of-the-art results on OpenImages dataset. 

<!-- ### Challenges in Partial Labeling
(a) In a partially labeled dataset, only a portion of the samples are annotated for a given class. (b) "Ignore" mode exploits only the annotated samples which may lead to a limited decision boundary. (c) "Negative" mode naively treats all un-annotated labels as negatives. It may produce suboptimal decision boundary as it adds noise of un-annotated positive labels. Also, annotated and un-annotated negative samples contribute similarly to the optimization. (d) Our approach aims at mitigating these drawbacks by predicting the probability of a label being present in the image.
 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/intro_modes_CSL.png" align="center" width="1000" ></td>
  </tr>
</table>
</p> -->

### Direct OpenImages Download is Now Available. 
We provide direct and convenient access for the OpenImages (V6) dataset. This will enable a common and reproducible baseline for benchmarking and future research. See further details [here](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/OpenImages.md).

### Class-aware Selective Approach
An overview of our approach is summarized in the following figure:
 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/CSL_approach.png" align="center" width="900" ></td>
  </tr>
</table>
</p>

### Loss Implementation 

Our loss consists of a selective approach that adjusts the training mode for each class individually and a partial asymmetric loss.
<!-- The selective approach is based on two probabilities quantities: label likelihood and label prior. The partial asymmetric loss emphasizes the contribution of the annotated labels.   -->
An implementation of the Class-aware Selective Loss (CSL) can be found [here](/src/loss_functions/partial_asymmetric_loss.py). 
- ```class PartialSelectiveLoss(nn.Module)```


## Pretrained Models
<!-- In this [link](MODEL_ZOO.md), we provide pre-trained models on various dataset.  -->
We provide models pretrained on the OpenImages dataset with different partial training-modes and architectures:

| Model | Architecture | Link | mAP |
| :---            | :---:      | :---:     | ---: |
| Ignore          | TResNet-M | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/mtresnet_opim_ignore.pth)      | 85.38       |
| Negative        | TResNet-M | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/mtresnet_opim_negative.pth)    | 85.85       |
| Selective (CSL) | TResNet-M  | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/mtresnet_opim_86.72.pth)      | 86.72       |
| Selective (CSL) | TResNet-L  | [link](https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/CSL/opim_v6/ltresnet_v2_opim_87.34.pth)   | **87.34**   |
 


## Inference Code (Demo)
We provide [inference code](infer.py), that demonstrates how to load the
model, pre-process an image and do inference. An example run of
OpenImages model (after downloading the relevant model):
```
python infer.py  \
--dataset_type=OpenImages \
--model_name=tresnet_m \
--model_path=./models_local/mtresnet_opim_86.72.pth \
--pic_path=./pics/10162266293_c7634cbda9_o.jpg \
--input_size=224
```

### Result Examples 
<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="./pics/demo_examples.png" align="center" width="900" ></td>
  </tr>
</table>
</p>



## Training Code
Training code is provided in [train.py](train.py). Also, code for simulating partial annotation for the [MS-COCO dataset](https://cocodataset.org/#download) is available ([coco_simulation](src/helper_functions/coco_simulation.py)). In particular, two "partial" simulation schemes are implemented: fix-per-class(FPC) and random-per-sample (RPS).
- FPC: For each class, we randomly sample a fixed number of positive annotations and the same number of negative annotations.  The rest of the annotations are dropped.
- RPS: We omit each annotation with probability p.

Pretrained weights using the ImageNet-21k dataset can be found here: [link](https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/MODEL_ZOO.md)\
Pretrained weights using the ImageNet-1k dataset can be found here: [link](https://github.com/Alibaba-MIIL/TResNet/blob/master/MODEL_ZOO.md)

Example of training with RPS simulation:
```
--data=/datasets/COCO/COCO_2014
--model-path=models/pretrain/mtresnet_21k
--gamma_pos=0
--gamma_neg=1
--gamma_unann=4
--simulate_partial_type=rps
--simulate_partial_param=0.5
--partial_loss_mode=selective
--likelihood_topk=5
--prior_threshold=0.5
--prior_path=./outputs/priors/prior_fpc_1000.csv
```

Example of training with FPC simulation:
```
--data=/mnt/datasets/COCO/COCO_2014
--model-path=models/pretrain/mtresnet_21k
--gamma_pos=0
--gamma_neg=3
--gamma_unann=4
--simulate_partial_type=fpc
--simulate_partial_param=1000
--partial_loss_mode=selective
--likelihood_topk=5
--prior_threshold=0.5
--prior_path=./outputs/priors/prior_fpc_1000.csv
```

### Typical Training Results

#### FPC (1,000) simulation scheme:
| Model                    | mAP         | 
| :---                     | :---:       |
| Ignore, CE               | 76.46       |
| Negative, CE             | 81.24       |
| Negative, ASL (4,1)      | 81.64       |
| CSL - Selective, P-ASL(4,3,1)  | **83.44**       |     


#### RPS (0.5) simulation scheme:
| Model                    | mAP        | 
| :---                     | :---:      |
| Ignore, CE               | 84.90      |
| Negative, CE             | 81.21      |
| Negative, ASL (4,1)      | 81.91      |
| CSL- Selective, P-ASL(4,1,1)  | **85.21**      |  


## Estimating the Class Distribution
The training code contains also the procedure for estimating the class distribution from the data. Our approach enables us to rank the classes based on predictions of a temporary model trained using the *Ignore* mode. [link](https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/cadc2afab73294a0e9e0799eec06b095e50e646e/src/loss_functions/partial_asymmetric_loss.py#L131)

#### Top 10 classes:
| Method                    | Top 10 ranked classes      | 
| :---                     | :---:      |
| Original                           | 'person', 'chair', 'car', 'dining table', 'cup', 'bottle', 'bowl', 'handbag', 'truck', 'backpack'   |
| Estiimate (Ignore mode)            | 'person', 'chair', 'handbag', 'cup', 'bench', 'bottle', 'backpack', 'car', 'cell phone', 'potted plant'      |
| Estimate (Negative mode)           | 'kite' 'truck' 'carrot' 'baseball glove' 'tennis racket' 'remote' 'cat' 'tie' 'horse' 'boat'    |


## Citation
```
@misc{benbaruch2021multilabel,
      title={Multi-label Classification with Partial Annotations using Class-aware Selective Loss}, 
      author={Emanuel Ben-Baruch and Tal Ridnik and Itamar Friedman and Avi Ben-Cohen and Nadav Zamir and Asaf Noy and Lihi Zelnik-Manor},
      year={2021},
      eprint={2110.10955},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Several images from [OpenImages dataset](https://storage.googleapis.com/openimages/web/index.html) are used in this project.
Some components of this code implementation are adapted from the repository https://github.com/Alibaba-MIIL/ASL.
