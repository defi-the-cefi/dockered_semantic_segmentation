# Semantic Segmentation Dockered MicroService

Semantic Segmentation as a docker microservice


## Overview
  * [Intro](#intro)
  * [Requirements](#requirements)
  * [Usage](#usage)
  * [References](#references)


### Overview
Docker packaged semantic segmentation CNN. Drop your images into the images folder, hit run and get back your images with corresponding semantic segmentation annotations. Modularized for easy integreation into larger systems that can reason and act on the visual semantic object information detected. Trivializes horizontal scalability across services such as AWS and Kubernetes.


#### Sample Output

Semantic map overlaid on original image. Mask are for Person, Car, Bike, Bus, Train, Boat, Motorbike, Dog, Cat

![sample_output](sample/person_and_car_semantic_segmentationsmall.png)



![gur_maths]()


### Requirements

  * Ubuntu 18+ OS
  * Docker
  * CUDA 10+ and 8gb+ gpu memory (optional, but highly recommended)

### Usage

  * git clone this repo
  ```
  cd ./dockered_semantic_segmentation
  ```
  * build docker image
  ```
  docker build -f Dockerfile -t seg_deep .
  ```
  * copy images for annotation to ./images folder
  ```
  cp /image/source/path ./images
  ```
  * launch and run container
  ```
  docker run --gpus all -u $(id -u):$(id -g) --name seg -it seg_deep bash
  ```
  * annotations and reshaped images to match annotations tensor shape can be found in the directory ./output_images


### References

[DeepLab]()

[Resnet]()
