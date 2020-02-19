# Violence Detection with C3D

This project implements a 3D Convolutional Neural Network (3D-CNN) for detecting violent scenes in a video stream. The 3D-CNN is a deep supervised learning approach that learns spatiotemporal discriminant features from videos (sequence of image frames). In contrast to 2D convolutions, this approach operates 3D kernels on a series of image frames in their context producing 3D activation maps that captures both spatial and temporal features that could not be properly identified with 2D convolutions.

### Model Training



### Pipeline

Each stage in the project pipeline is displayed in the image below:



* Pre-Processing: To avoid overworking the inference engine, the pretrained MobileNet model (from the Intel OpenVINO Model Zoo) is used to detect persons in a given frame. When a person is detected, a stack of 16 frames are collected and passed through the 3D-CNN model to detect violence.

* Violence Detection: The

### Datasets

The links to the datasets used are

* [Hockey Fight](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech)
* [Movies](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635)
* [Crowd Violence](https://www.openu.ac.il/home/hassner/data/violentflows/)

### Results


### Further Steps

* Train the model to better accuracy by searching for the best hyperparameters and optimizer
* Implement multi-camera feed system
* Build complete web app for visualization
* Implement geolocation for even violence event