# Real-time Age and Gender Estimation using OpenCV

👨‍🔬🔍 This Python script uses OpenCV's deep neural networks to perform real-time age and gender estimation from webcam or video input.

## Requirements

- Python 3.x
- OpenCV (cv2)
  
## Installation

1. Install Python if you haven't already. You can download it from the [official Python website](https://www.python.org/).
2. Install OpenCV by running `pip install opencv-python`.

## Usage

1. Download the pre-trained models from the links below and place them in the same directory as the script:
   - [Face detection model (`opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt`)](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
   - [Age estimation model (`age_net.caffemodel` and `age_deploy.prototxt`)](https://github.com/GilLevi/AgeGenderDeepLearning)
   - [Gender estimation model (`gender_net.caffemodel` and `gender_deploy.prototxt`)](https://github.com/GilLevi/AgeGenderDeepLearning)

2. Run the script by executing `python real_time_age_gender_estimation.py`.

3. The webcam feed will open up showing real-time age and gender estimation of faces detected.

4. Press 'q' to quit the application.

## Models

1. **Face Detection Model:**
   - Responsible for detecting faces in images or video frames.
   - Files: `opencv_face_detector_uint8.pb` and `opencv_face_detector.pbtxt`.
   - Trained using Convolutional Neural Networks (CNNs) on a large dataset of face images.

2. **Age Estimation Model:**
   - Predicts the age group of the detected faces.
   - Files: `age_net.caffemodel` and `age_deploy.prototxt`.
   - Trained on a dataset where faces are labeled with age groups.

3. **Gender Estimation Model:**
   - Predicts the gender (male or female) of the detected faces.
   - Files: `gender_net.caffemodel` and `gender_deploy.prototxt`.
   - Trained on a dataset containing labeled face images with corresponding gender labels.

## References

- [OpenCV](https://opencv.org/)
- [Age and Gender Classification using Convolutional Neural Networks](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf)

## Credits

This script is based on the work by [Gil Levi and Tal Hassner](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf) on age and gender classification using convolutional neural networks.

