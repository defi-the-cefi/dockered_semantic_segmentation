# Semantic Segmentation Dockered MicroService

Gated Recurrent Network (GRU) to GRUow your savings

## Overview
  * [Intro](#intro)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [Results](#results)
  * [Dex Swap](#dexswap)
  * [References](#references)


### Overview
Docker packaged semantic segmentation CNN. 


#### Sample Output

Semantic map overlaid on original image. Mask are for Person, Car, Bike, Bus, Train, Boat, Motorbike, Dog, Cat

![sample_output](sample/person_and_car_semantic_segmentationsmall.png)

Below is the GRU circuit's math, i.e. the above circuit in the form of math equations whose parameters we will train to estimate

![gur_maths]()


### Requirements
  * Python 3.8
  * matplotlib == 3.1.1
  * numpy == 1.19.4
  * pandas == 0.25.1
  * torch == 1.11.0

Python pakcage dependencies can be installed using the following command:
```
pip install -r requirements.txt
```
Optional - For training on a GPU (highly recommended), Nvidia CUDA 10.0+ drivers are required

### Usage

### References

[DeepLab]()

[Resnet]()
