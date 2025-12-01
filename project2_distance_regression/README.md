📏 Project 2 — Depth Regression from Single Images

Machine Learning for Computer Vision · ETH Zürich · 2025

This project implements a convolutional regression model to estimate
camera-to-obstacle distance from a single RGB image.

It was part of the ETH Zürich Machine Learning for Computer Vision course (2025).

🔍 Overview

The task:
Predict the distance (in meters) to the nearest obstacle using only a single input image.

The model was trained and evaluated on the official ETH dataset.
Performance on the leaderboard is measured using Mean Absolute Error (MAE) in meters.

🧠 Model & Training Pipeline

Full ML pipeline implemented:

dataset preprocessing

normalization

augmentation

training / validation split

Convolutional neural network for continuous depth prediction

Loss function: MAE

Evaluation metric: MAE (meters)

Framework: PyTorch

🏗 Architecture

Custom CNN with:

convolution → ReLU → pooling

deeper layers with batch normalization

fully connected regression head

Outputs a single scalar distance value
