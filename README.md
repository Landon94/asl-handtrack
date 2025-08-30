# asl-handtrack

## Overview

## Dataset

This project uses the **Synthetic ASL Alphabet** dataset by [Lexset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet).  
It contains 27,000 synthetic images of American Sign Language alphabet signs generated with Lexset's Seahaven platform.

## CNN Architecture

### Conv2D
- **How it Works:** Slides a K×K kernel across the input, computing dot products to produce a feature map. 
- **Purpose:** Extracts features from images

### MaxPooling2D
- **How it Works:** Slides a K×K window across the feature map and keeps only the maximum value from each region.  
- **Purpose:** Decreases computational complexity and increases speed of the network

### Dropout 
- **How it Works**: Randomly sets a fraction of neurons to zero during training, so network does not become overly depedent on certain nodes
- **Purpose**: Reduces overfitting

### Flatten
- **How it Works:** Converts a multi-dimensional matrix into a 1D vector.  
- **Purpose:** Bridges the convolutional layers with the fully connected layers

### Dense
- **How it Works:** Computes the dot product of the inputs and weights, adds a bias term, and then applies an activation function.
- **Purpose:** Makes final prediction or classifaction for network

## Model Evaluation

### Confusion Matrix
<img width="3322" height="2921" alt="Image" src="https://github.com/user-attachments/assets/6a8bb303-07cd-4ba1-992d-988d720e4768" />

### Classification Report
<img width="573" height="611" alt="Image" src="https://github.com/user-attachments/assets/aff6ecd0-706d-4137-8a48-a3c961d2a597" />
