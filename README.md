# Superman_Batman_Detection_on_Yolov5_Using_Jetson_Nano_2GB

## Aim
The Superman-Batman Detection system is designed to identify and classify images of Superman and Batman, in which, it will detect Superman 'S' symbol and Batman mask.

## Objectives
• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether the object is Superman 'S' symbol or Batman mask.

• To Show users how to use the trained model to make predictions on new images, including specifying inference parameters and interpreting the output results.

## Abstract
• The Superman-Batman Detection System using YOLOv5 is an advanced object detection framework designed to accurately identify and classify images containing Superman and Batman.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• Users can effectively implement the Superman-Batman Detection System, enabling various applications such as automated content filtering, security surveillance, and entertainment.

## Introduction
• This project is based on a Superman-Batman detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.

• The Superman-Batman Detection System using YOLOv5 aims to accurately identify and classify images containing Superman and Batman. 

• YOLOv5 (You Only Look Once, version 5) is a leading object detection algorithm known for its real-time detection capabilities and high accuracy. 

• This documentation provides a comprehensive guide to help users set up, train, and use the system effectively. 

• Neural networks and machine learning have been used for these tasks and have obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Helmet detection as well.

• The system leverages YOLOv5 for detecting and classifying images. The model can process images quickly and deliver precise results, making it suitable for applications requiring real-time object detection, such as automated content filtering, security surveillance, and entertainment.

## Literature Review
• Object detection is a fundamental task in computer vision that involves identifying and localizing objects within an image. 

• It has wide-ranging applications, from surveillance and security to autonomous driving and medical imaging. 

• The field has seen significant advancements with the advent of deep learning, particularly Convolutional Neural Networks (CNNs).

Historical Context and Evolution of Object Detection
-
1) R-CNN (Region-based Convolutional Neural Networks) :
    Introduced by Girshick et al. in 2014, R-CNN utilizes selective search to generate region proposals and applies a CNN to each proposal for classification and bounding box regression. Despite its accuracy, R-CNN is computationally expensive and slow due to multiple CNN evaluations per image.

2) Fast R-CNN :
    Proposed by Ross Girshick in 2015, Fast R-CNN improved upon R-CNN by sharing convolutional computations across proposals. It introduced ROI (Region of Interest) pooling to speed up the process. However, it still relied on selective search for generating region proposals.

3) Faster R-CNN :
    Introduced by Ren et al. in 2015, Faster R-CNN integrated Region Proposal Networks (RPN) with Fast R-CNN, eliminating the need for selective search. This integration significantly improved both speed and accuracy, establishing Faster R-CNN as a benchmark in object detection.

4) YOLO (You Only Look Once) :
    Introduced by Redmon et al. in 2016, YOLO reframed object detection as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation. YOLO's primary advantage is its real-time detection speed. However, its initial versions had limitations in detecting smaller objects and achieving high accuracy compared to R-CNN variants.

5) YOLOv3 and YOLOv4 :
    YOLOv3 and YOLOv4 brought significant improvements in architecture and performance. YOLOv3 introduced multi-scale predictions and better backbone networks, while YOLOv4 incorporated several enhancements like CSPDarknet53 backbone, PANet path-aggregation neck, and CIoU loss for better bounding box regression.

YOLOv5: Advancements and Features
-
YOLOv5, developed by Ultralytics, represents the latest iteration in the YOLO series, offering several improvements over its predecessors:

• Architecture: YOLOv5 uses CSP (Cross Stage Partial) networks, enhancing feature extraction and detection capabilities.
• Training: Simplified training process with built-in support for data augmentation, mixed precision training, and automatic hyperparameter tuning.
• Performance: Maintains a balance between speed and accuracy, making it suitable for real-time applications.

YOLOv5's user-friendly implementation and state-of-the-art performance have made it a popular choice for various object detection tasks.

## Jetson Nano Compatibility
• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Jetson Nano 2gb

## Proposed System
1] Study basics of machine learning and image recognition.

2]Start with implementation

• Front-end development

• Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a Helmet or not.

4] use datasets to interpret the object and suggest whether the person on the camera’s viewfinder is wearing a helmet or not.

## Methodology
The Helmet detection system is a program that focuses on implementing real time Helmet detection. It is a prototype of a new product that comprises of the main module: Helmet detection and then showing on viewfinder whether the person is wearing a helmet or not. Helmet Detection Module

This Module is divided into two parts:

1] Head detection
  • Ability to detect the location of a person’s head in any input image or frame. The output is the bounding box coordinates on the detected head of a person.
  • For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.
  • This Datasets identifies person’s head in a Bitmap graphic object and returns the bounding box image with annotation of Helmet or no Helmet present in each image.

2] Helmet Detection
  • Recognition of the head and whether Helmet is worn or not.
  • Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.
  • There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
  • YOLOv5 was used to train and test our model for whether the helmet was worn or not. We trained it for 149 epochs and achieved an accuracy of approximately 92%. 

## Installation
Initial Configuration

sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*

Create Swap
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0

Cuda env in bashrc
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

Update & Upgrade
sudo apt-get update
sudo apt-get upgrade

Install some required Packages
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow

Install Torch
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"

Install Torchvision
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install

Clone Yolov5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt


Download weights and Test Yolov5 Installation on USB webcam
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0

## Helmet Dataset Training
# We used Google Colab And Roboflow
train your model on colab and download the weights and past them into yolov5 folder link of project

colab file given in repo

## Running Helmet Detection Model
source '0' for webcam
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

## demo

https://github.com/priyankapatil2345/superman_batman-detection/assets/147481327/c437bf5b-6972-4dc7-b360-e4236b0a1756

### Link:- https://youtu.be/OHsNWNB1-jY?si=_-1-xexuKYeuG6Gn

## Advantages
• Helmet detection system will be of great help in minimizing the injuries that occur due to an accident.

• Helmet detection system shows whether the person in viewfinder of camera module is wearing a Helmet or not with good accuracy.

• It can then convey it to authorities like traffic policeman or the data about the respective person and his vehicle can be stored, and then based on the info acquired can be notified on his mobile phone about the Helmet using law.

• When completely automated no user input is required and therefore works with absolute efficiency and speed.

• It can work around the clock and therefore becomes more cost efficient.

## Application
• Detects a person’s head and then checks whether Helmet is worn or not in each image frame or viewfinder using a camera module.

• Can be used anywhere where traffic lights are installed as their people usually stop on red lights and Helmet detection becomes even more accurate.

• Can be used as a reference for other ai models based on Helmet Detection.

## Future Scope
• As we know technology is marching towards automation, so this project is one of the step towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

• Helmet detection will become a necessity in the future due to rise in population and hence our model will be of great help to tackle the situation in an efficient way.

## Conclusion
• In this project our model is trying to detect a person’s head and then showing it on viewfinder, live as to whether Helmet is worn or not as we have specified in Roboflow.

• The model tries to solve the problem of severe head injuries that occur due to accidents and thus protects a person’s life.

• The model is efficient and highly accurate and hence reduces the workforce required.

## Reference
1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/search/2/image?phrase=helmet

3] Google images

## Articles :-
1] https://www.bajajallianz.com/blog/motor-insurance-articles/what-is-the-importance-of-wearing-a-helmet-while-riding-your-two-wheeler.html#:~:text=Helmet%20is%20effective%20in%20reducing,are%20not%20wearing%20a%20helmet.

2] https://www.findlaw.com/injury/car-accidents/helmet-laws-and-motorcycle-accident-cases.html





