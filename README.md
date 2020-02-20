# Violence Detection with C3D

This project implements a 3D Convolutional Neural Network (3D-CNN) for detecting violent scenes in a video stream. The 3D-CNN is a deep supervised learning approach that learns spatiotemporal discriminant features from videos (sequence of image frames). In contrast to 2D convolutions, this approach operates 3D kernels on a series of image frames in their context producing 3D activation maps that captures both spatial and temporal features that could not be properly identified with 2D convolutions.

## System Pipeline

<img src="images/pipeline.png" width="100%" heigth="100%">
Image source: [1]

Each stage in the project pipeline is displayed in the image below:

* *Pre-Processing:* To avoid overworking the inference engine, the pretrained MobileNet model (from the Intel OpenVINO Model Zoo) is used to detect persons in a given frame. When a person is detected, a stack of 16 frames are collected and passed through the 3D-CNN model to detect violence.

* *Violence Detection:* Each 16 frame sequence is passed throught the trained 3D CNN model which outputs whether the scene is violent or not as probability scores. The class with maximum output is the predicted value.

* *Visualization:* A front-end interface allows the operation of the system to be viewed in real-time. The video frame is played on-screen with an indicator area that flags violent scenes.

* *Alert:* When violence is detected in any of the scenes, nearest security group is notified for immediate response.

## Model Training

The 3D CNN model was custom trained using the architecture shown below:

### Datasets

Three datasets were combined for this task: [Hockey Fight](http://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89/tech), [Movies](http://academictorrents.com/details/70e0794e2292fc051a13f05ea6f5b6c16f3d3635) and [Crowd Violence](https://www.openu.ac.il/home/hassner/data/violentflows/).

The Hockey Fight dataset contained 1000 video clips with half containing violent scenes and the other non-violent. The Movies dataset contained 200 video clips with half containing violent scenes and the other non-violent. The Crowd violence dataset contained 246 video clips from YouTube with half containing violent scenes and the other non-violent. This gave a total of 1446 videos, with 723 videos each violent and non-violent.

Image frames are extracted from these videos using the script in `/data/video2img.sh` (gotten from [JJBOY](https://github.com/JJBOY/C3D-pytorch)) at a sampling rate of 16 frames per second. This value was chosen arbitrarily and is good enough for a start. The different image frames are then collected into stacks with 16 frames per stack using `/data/create_stacks.py` using information provided in the `/data/train.txt` and `/data/test.txt` that specifies the starting point of each stack. This was necessary as the frames were sent in overlapping sequences.

The entire dataset was divided into training set and test set in the ratio 3:1. This was according to the method in [1]. Both sets were then bundled into the HDF5 format using the `h5py` package.

### Data pre-processing

Each image stack is first resized to 128 by 171 pixels before they are square cropped to 112 by 112 px according to the input shape of the 3D CNN. They are then converted to PyTorch tensors and each of the RGB frames normalized with `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`. This is a common transformation practice that is derived from the normalization introduced by ImageNet.

### Hyperparameters and Optimization

For the training task, the following hyperparameters were used:

* `num_epochs    = 100`
* `batch_size    = 30`
* `learning_rate = 0.003`

These parameters are in no way optimal, but gave fairly good result for a start.

The stochastic gradient descent optimizer was used for learning with Cross Entropy function was used as the criterion for classification loss.

### Training Setup

The training and test datasets were uploaded to Google Drive since all training was done in Google Colab. Colab is a free cloud service that gives access to free GPU instances and supports most known libraries including PyTorch that was used for this project. The training environment has the following specifications:

* GPU: 1xTesla K80 ,compute 3.7, having 2496 CUDA cores, 12GB GDDR5 VRAM
* CPU: 1xsingle core hyper threaded Xeon Processors @2.3Ghz i.e. (1 core, 2 threads)
* RAM: ~12.6 GB Available
* Disk: ~33 GB Available

## Model Results

### Test Loss

<img src="images/loss_graph.png" width="100%" heigth="100%">

### Test Accuracy

<img src="images/accuracy_graph.png" width="100%" heigth="100%">

The best accuracy of 84.06% was obtained at the 36th training epoch.

This preliminary model is definitely far from the best. Apparently, the model was subject to overfiiting and a lot of improvement can be achieved with proper training.

## Edge Inferencing

## Further Steps

* Train the model to better accuracy by searching for the best hyperparameters and optimizer
* Implement multi-camera feed system
* Build complete web app for visualization
* Implement geolocation for improved violence event reporting

## References

[1] Ullah, F. U. M., Ullah, A., Muhammad, K., Haq, I. U., & Baik, S. W. (2019). Violence Detection Using Spatiotemporal Features with 3D Convolutional Neural Network. *Sensors*, 19(11), 2472. https://doi.org/10.3390/s19112472

[2] Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning Spatiotemporal Features with 3D Convolutional Networks. *2015 IEEE International Conference on Computer Vision (ICCV)*. https://doi.org/10.1109/iccv.2015.510