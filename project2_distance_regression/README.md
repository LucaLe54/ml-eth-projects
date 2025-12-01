📏 Project 2 — Depth Regression from Single Images

ETH Zürich · Machine Learning for Computer Vision (2025)

This project trains a regression model to estimate the distance from the camera to the nearest obstacle using only a single low-resolution DSF16 image.

🔍 Goal

Predict a continuous depth value (in meters) from a single grayscale image.
The task was evaluated using Mean Absolute Error (MAE) on a leaderboard.

🧠 Method

This project uses a classical machine-learning approach:

Custom preprocessing (Box-Cox + StandardScaler)

Data augmentation (90° rotations)

Ensemble model:

k-Nearest Neighbors

Kernel Ridge Regression

Combined using VotingRegressor

Hyperparameter tuning with GridSearchCV

🏗 Architecture

Input: 27×36 grayscale images (flattened)

Preprocessing: Power transform → Scaling

Models:

kNN (distance weighting)

Kernel Ridge with RBF kernel

Output: Single scalar distance value

📁 Files

train.py — training, evaluation, test prediction
