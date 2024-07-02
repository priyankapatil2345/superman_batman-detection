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

• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Superman-Batman detection as well.

• The system leverages YOLOv5 for detecting and classifying images. The model can process images quickly and deliver precise results, making it suitable for applications requiring real-time object detection, such as automated content filtering, security surveillance, and entertainment.

## Literature Review
• Object detection is a fundamental task in computer vision that involves identifying and localizing objects within an image. 

• It has wide-ranging applications, from surveillance and security to autonomous driving and medical imaging. 

• The field has seen significant advancements with the advent of deep learning, particularly Convolutional Neural Networks (CNNs).

### Historical Context and Evolution of Object Detection

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
The proposed system leverages YOLOv5 to detect and classify images containing Superman and Batman. YOLOv5’s real-time detection capabilities and high accuracy make it suitable for this application. The system will be designed to handle various tasks, including data preparation, model training, inference, and deployment.

## Methodology
# Workflow
1) Data Collection and Annotation :
•  Collect a diverse dataset of Superman and Batman images.
•  Annotate the images using LabelImg or a similar tool.

2) Data Preparation :
•  Organize images and labels into the required directory structure.
•  Create and configure the data.yaml file.

3) Training the Model :
•  Run the YOLOv5 training script with appropriate parameters.
•  Monitor training progress and adjust parameters if necessary.

4) Running Inference :
•  Use the trained model to detect Superman and Batman in new images.
•  Adjust confidence thresholds and other parameters for optimal detection.

5) Evaluating the Model :
•  Run the validation script to evaluate model performance.
•  Analyze metrics and make adjustments if needed.

6) Deploying the Model :
•  Export the trained model in desired formats for integration into various applications.

# Potential Challenges and Solutions
1) Data Quality and Quantity
•  Challenge: Insufficient or low-quality images may affect model performance.
•  Solution: Augment the dataset with synthetic images or use transfer learning to enhance model training.

2) Model Overfitting
•  Challenge: The model may overfit to the training data, reducing generalization.
•  Solution: Use regularization techniques and data augmentation to improve generalization.

3) Inference Speed
•  Challenge: Ensuring real-time detection performance.
•  Solution: Optimize model parameters and use hardware acceleration (e.g., GPU).

The Superman-Batman Detection System using YOLOv5 aims to provide accurate, real-time detection of these characters for applications in automated content filtering, security surveillance, and entertainment. By leveraging YOLOv5’s capabilities and following a structured approach, the system ensures high performance and reliability.

## Installation
# Prerequisites
1. Python: Ensure Python 3.8 or higher is installed.
2. CUDA: For GPU support, install CUDA and cuDNN.
3. PIP: Ensure you have the latest version of pip.

# Step-by-Step Installation
1. Set Up Python Environment
    It's recommended to use a virtual environment to manage dependencies:
 
        python -m venv yolo-env
        source yolo-env/bin/activate  # On Windows use `yolo-                    env\Scripts\activate`

2. Install PyTorch and Torchvision

Follow the instructions from the PyTorch website to install PyTorch and Torchvision. For example, if you have CUDA 11.3, use:

        pip install torch torchvision torchaudio --index-url                     https://download.pytorch.org/whl/cu113

3. Clone the YOLOv5 Repository

Clone the official YOLOv5 repository from GitHub:

        git clone https://github.com/ultralytics/yolov5.git
        cd yolov5

4. Install YOLOv5 Dependencies

Install the required dependencies:

        pip install -r requirements.txt

5. Set Up the Dataset

Create a directory structure for your dataset:

        mkdir -p dataset/images/train
        mkdir -p dataset/images/val
        mkdir -p dataset/labels/train
        mkdir -p dataset/labels/val

Place your annotated images and label files in the respective directories.

6. Create the Configuration File

In the YOLOv5 directory, create a data.yaml file:

        train: ../dataset/images/train
        val: ../dataset/images/val
        nc: 2
        names: ['Superman', 'Batman']

7. Train the Model

Start training the model using the YOLOv5 training script:

        python train.py --img 640 --batch 16 --epochs 50 --data                  data.yaml --cfg yolov5s.yaml --weights yolov5s.pt

• --img: Image size.
• --batch: Batch size.
• --epochs: Number of epochs.
• --data: Path to your data.yaml file.
• --cfg: Model configuration (e.g., yolov5s for the small model).
• --weights: Pre-trained weights to start from (e.g., yolov5s.pt).

8) Inference and Detection

After training, use the trained model for inference:

        python detect.py --weights runs/train/exp/weights/best.pt --img 
        640 --conf 0.25 --source path/to/your/image.jpg

• --weights: Path to the trained model weights.
• --img: Image size.
• --conf: Confidence threshold for detections.
• --source: Source image or directory of images.

9) Model Evaluation

Evaluate the model’s performance using the validation dataset:

        python val.py --weights runs/train/exp/weights/best.pt --data            data.yaml --img 640

10) Export the Model for Deployment

Export the trained model in various formats for deployment:
      
        python export.py --weights runs/train/exp/weights/best.pt --img          640 --include onnx coreml tflite

• --include: Formats to export (e.g., onnx, coreml, tflite).


By following these installation steps, you can set up the Superman-Batman detection system using YOLOv5. This setup includes creating a Python virtual environment, installing necessary dependencies, setting up your dataset, training the model, running inference, evaluating the model, and exporting it for deployment.

## Demo

https://github.com/priyankapatil2345/superman_batman-detection/assets/147481327/c437bf5b-6972-4dc7-b360-e4236b0a1756

### Link:- https://youtu.be/OHsNWNB1-jY?si=_-1-xexuKYeuG6Gn

## Advantages
1) Real-Time Detection:
• Speed: Optimized for fast, real-time processing.
• Efficiency: Single pass detection ensures quick responses.

2) High Accuracy:
• Precision: Minimizes false positives and negatives.
• Recall: Effective in complex environments and varying conditions.

3) Scalability:
• Dataset Flexibility: Handles large, diverse datasets.
• Model Configurations: Adaptable to different computational resources and needs.

4) Ease of Use:
• Pre-trained Weights: Accelerates training with pre-trained models.
• User-Friendly: Well-documented and easy-to-use training scripts.

5) Adaptability:
• Transfer Learning: Customizable for specific detection tasks.
• Data Augmentation: Enhances model robustness through simulated scenarios.

## Application
1) Entertainment:
• AR and Games: Enhance interactive experiences.
• Video Editing: Automate tagging for easier editing.

2) Security and Surveillance:
• Event Monitoring: Identify characters for security.
• Crowd Management: Improve safety and organization.

3) Content Management:
• Filtering: Automate content filtering for copyright compliance.
• Metadata: Generate tags for better media organization.

4) Marketing and Advertising:
• Targeted Ads: Trigger ads based on character detection.
• Interactive Displays: Use in public spaces for engagement.

5) Retail and Merchandise:
• Inventory Management: Track Superman and Batman merchandise.
• Customer Interaction: Enhance experiences with character recognition.

## Future Scope
• More Characters: Expand the model to detect additional superheroes and characters.

• Dynamic Updates: Continuously update the dataset with new images and scenarios to improve model accuracy.

• User Behavior Analysis: Analyze user interactions with detected characters for personalized marketing strategies.

• Dynamic Content: Develop systems that alter content in real-time based on character detection to enhance user engagement.

• As we know technology is marching towards automation, so this project is one of the step towards automation.

• Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.

• The future scope of Superman-Batman detection using YOLOv5 is vast, with opportunities for improvement and integration into various fields, from entertainment and security to healthcare and beyond.

## Conclusion
• Implementing a Superman-Batman detection system using YOLOv5 provides a powerful, real-time solution for identifying these characters in various applications.

• The approach leverages YOLOv5's speed, accuracy, and scalability, making it suitable for entertainment, security, content management, marketing, and beyond. 

• The future scope includes extending detection capabilities, improving performance, integrating across platforms, and exploring new applications in AR, VR, gaming, and healthcare. 

• This system represents a significant advancement in object detection technology, offering broad and impactful uses.

## Reference
1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.gettyimages.ae/search/2/image?phrase=superman%20batman&sort=mostpopular&license=rf%2Crm

3] Google images

## Articles :-
1] https://www.kaggle.com/datasets/mralamdari/superman-vs-batman-vs-person.

2] https://medium.com/@mralamdari/detect-superman-by-yolo-5d81a065a95e. 

3] https://www.kaggle.com/datasets/mralamdari/superman-or-batman.





