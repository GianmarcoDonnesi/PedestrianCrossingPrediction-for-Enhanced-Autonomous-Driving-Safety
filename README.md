# Pedestrian Crossing Prediction using CNN-LSTM Networks for Enhanced Autonomous Driving Safety

<img src="https://github.com/GianmarcoDonnesi/ComputerVisionProject/blob/main/cover.jpg" alt="Example Image" width="700"/>

## Overview
This project focuses on predicting pedestrian crossing behavior using the *JAAD 2.0* (Joint Attention for Autonomous Driving) dataset. The project pipeline involves extracting clips from the JAAD dataset, compute bounding boxes and extract keypoints from the images with *Mediapipe*, preprocessing annotations, creating datasets for training and evaluation, and training a deep learning model based on VGG19 and LSTM with attention mechanisms.

### Abstract
*Pedestrian intention estimation is an open key challenge in assistive and autonomous driving systems for urban environments. An intelligent system should be able to understand the intentions of pedestrians and predict their forthcoming actions in real-time. To date, only a few public datasets have been proposed for the purpose of studying pedestrian behavior prediction in the context of intelligent driving. This project leverages the JAAD dataset, which contains annotated video sequences of pedestrian behaviors, to develop a simplified yet effective model for predicting pedestrian crossing behavior leveraging computer vision techniques. Using bounding box annotations to track pedestrian movements across frames of the dataset video sequences, our approach employs a modified version of a pre-trained Convolutional Neural Network (CNN) followed by a Long Short-Term Memory (LSTM) network to analyze spatial and temporal patterns. The model outputs a binary prediction-whether a pedestrian will cross the street within a given timeframe-offering a direct solution to the complex challenge of pedestrian intention prediction. We evaluate our model's performance using standard metrics such as accuracy, recall and F1-score. This project aims to provide a simplified and extendable version of a method that not only contributes to the enhancement of safety in intelligent autonomous driving systems, but also serves as an instructive example of applying Computer Vision and Deep Learning techniques to understand and predict real-world pedestrian behaviors.*

#### Authors
<p>
  Gianmarco Donnesi<br>
  Matr. n. 2152311
</p>

<p>
  Michael Corelli<br>
  Matr. n. 1938627
</p>

---

## Setup and Requirements
Before running the scripts, ensure you have all the necessary dependencies installed. This can be done by installing the requirements from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 1: Extract JAAD Clips
Run the script `extractJAAD_clips.py` to extract clips from the JAAD dataset.

### Step 2: Generate Bounding Boxes and Pose Keypoints
Run the script `bboxes_and_posekeypoints.py` to generate bounding boxes and pose keypoints for each frame in the video clips. It uses computer vision techniques, like *Mediapipe*, to detect and annotate pedestrian positions and movements.

### Step 3: Preprocess Annotations
Run the scritp `preprocess_annotations.py` to preprocess the raw annotations transforming them into a structured format suitable for training and evaluation.

### Step 4: Create Dataset
Run the script `create_dataset.py` to prepare the dataset for training and evaluation by combining frames, keypoints, and annotations.

### Step 5: Model Definition
The `model/model.py` script contains the definition of the model used for prediction. <br>
This script defines the `PedestrianCrossingPredictor` model, which uses a combination of *VGG19* for feature extraction and *LSTM* with attention mechanisms for sequence modeling.

### Step 6: Train and Validate the model
Run the script `model/train.py` to load the data, train the model for crosswalk prediction and finally save trained model for future use in the *trained_model.pth* file. <br>
Now you can use the script `model/validation.py` to evaluate the performance of the trained model on the validation dataset, an ablation analysis is also performed to understand the importance of different input features.

### Step 7: Model results and graphs
Run the script `model/graphs.py` to see graphs showing how different ablations affect the model's Accuracy, Recall and F1 Score. <br>
From these we understand the importance of different input features in predicting pedestrian crossings. <br>
The three graphs show for each type of ablation: Accuracy, Recall and F1 Score.

### Technologies and Tools Used

This project utilizes a variety of tools and libraries to achieve its goals:

- **Mediapipe**: A cross-platform framework used for building multimodal applied machine learning pipelines, particularly for extracting pose keypoints from video frames.
- **OpenCV**: An open-source computer vision and machine learning software library. Used for video frame processing and manipulation.
- **PyTorch**: An open-source machine learning library based on the Torch library, used for developing and training deep learning models.
- **NumPy**: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- **Scikit-Learn**: A machine learning library for Python, used for implementing machine learning algorithms and evaluation metrics like accuracy, recall, and F1 score.
- **JAAD Dataset**: The primary dataset used for training and evaluating the model, containing annotated video clips of pedestrians in urban environments.

---
## License
Unless otherwise stated, the code in this repository is licensed under [Apache-2.0 license](LICENSE). This permits reuse and distribution according to the terms of the license.
