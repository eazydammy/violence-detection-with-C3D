# Violence Detection with C3D

This project implements a 3D Convolutional Neural Network (3D-CNN) for detecting violent scenes in a video stream. The 3D-CNN is a deep supervised learning approach that learns spatiotemporal discriminant features from videos (sequence of image frames). In contrast to 2D convolutions, this approach operates 3D kernels on a series of image frames in their context producing 3D activation maps that captures both spatial and temporal features that could not be properly identified with 2D convolutions.

## System Pipeline

<img src="images/pipeline.png" width="100%" heigth="100%">

Each stage in the project pipeline is displayed in the image below:

* *Pre-Processing:* To avoid overworking the inference engine, the pretrained MobileNet model (from the Intel OpenVINO Model Zoo) is used to detect persons in a given frame. When a person is detected, a stack of 16 frames are collected and passed through the 3D-CNN model to detect violence.

* *Violence Detection:* Each 16 frame sequence is passed throught the trained 3D CNN model which outputs whether the scene is violent or not as probability scores. The class with maximum output is the predicted value.

* *Visualization:* A front-end interface allows the operation of the system to be viewed in real-time. The video frame is played on-screen with an indicator area that flags violent scenes.

* *Alert:* When violence is detected in any of the scenes, nearest security group is notified for immediate response.

## Model Training

The 3D CNN model was custom trained using the architecture shown below:

### Datasets

Three datasets were combined for this task: [Hockey Fight](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech), [Movies](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635) and [Crowd Violence](https://www.openu.ac.il/home/hassner/data/violentflows/).

Hocket


### Data pre-processing

### Hyperparameters and Optimization

### Training Setup

The links to the datasets used are



## Results

## Further Steps

* Train the model to better accuracy by searching for the best hyperparameters and optimizer
* Implement multi-camera feed system
* Build complete web app for visualization
* Implement geolocation for improved violence event reporting

## References

[1] Ullah, F. U. M., Ullah, A., Muhammad, K., Haq, I. U., & Baik, S. W. (2019). Violence Detection Using Spatiotemporal Features with 3D Convolutional Neural Network. *Sensors*, 19(11), 2472. https://doi.org/10.3390/s19112472

[2] Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning Spatiotemporal Features with 3D Convolutional Networks. *2015 IEEE International Conference on Computer Vision (ICCV)*. https://doi.org/10.1109/iccv.2015.510