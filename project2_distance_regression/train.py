"""
Depth Regression Model (ETH Zürich – ML for Computer Vision, 2025)
------------------------------------------------------------------
This script trains a regression model to estimate camera-to-obstacle
distance from a single image using classical machine-learning methods.

Model: VotingRegressor(kNN + Kernel Ridge)
Input: Flattened grayscale/DSF16 images (27×36)
Output: Distance in meters
"""

from utils import (
    load_config,
    load_dataset,
    load_test_dataset,
    save_results,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib

SHAPE = (27, 36)  # Dataset image resolution


def augment_with_rotation_90(images, distances, shape):
    """
    Simple augmentation: rotate each image by 90°.
    Keeps the regression label unchanged.
    """
    X_aug, y_aug = [], []
    for img, y in zip(images, distances):
        img2d = img.reshape(shape)

        rot = np.rot90(img2d, k=1)  # 90° rotation

        X_aug.append(img)
        X_aug.append(rot.flatten())
        y_aug.extend([y, y])

    return np.array(X_aug), np.array(y_aug)


if __name__ == "__main__":
    # Load config + dataset
    config = load_config()
    images, distances = load_dataset(config)
    print(f"[INFO] Loaded {len(images)} samples.")

    # Data augmentation
    images, distances = augment_with_rotation_90(images, distances, SHAPE)
    print(f"[INFO] Augmented dataset: {len(images)} samples.")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        images, distances, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    preprocessor = Pipeline([
        ("power", PowerTransformer(method="box-cox", standardize=False)),
        ("scaler", StandardScaler()),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    # Base models for the ensemble
    knn = KNeighborsRegressor()
    krr = KernelRidge()

    model = VotingRegressor(
        estimators=[("knn", knn), ("krr", krr)],
        weights=[7, 3],
    )

    # Hyperparameters
    param_grid = {
        "knn__n_neighbors": [2],
        "knn__weights": ["distance"],
        "knn__p": [2],
        "krr__alpha": [0.006, 0.0065, 0.007, 0.009, 0.005],
        "krr__kernel": ["rbf"],
        "krr__gamma": [0.0016, 0.0015, 0.0014, 0.0013],
    }

    # Grid search
    grid = GridSearchCV(
        model,
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"[INFO] Best parameters: {grid.best_params_}")
    print(f"[INFO] Best CV MAE: {-grid.best_score_:.4f} m")

    # Validation
    y_val_pred = best_model.predict(X_val)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    print(f"[INFO] Validation MAE: {mae_val:.4f} m")

    # Save full inference pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", best_model)
    ])
    joblib.dump(pipeline, "pipeline.joblib")
    print("[INFO] Saved inference pipeline to pipeline.joblib")

    # Predict test set
    X_test = np.array(load_test_dataset(config))
    X_test = preprocessor.transform(X_test)
    test_pred = best_model.predict(X_test)

    save_results(test_pred)
    print("[INFO] Predictions saved.")
