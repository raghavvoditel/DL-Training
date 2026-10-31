# K-Fold Cross Validation with a Keras ANN.
# Instead of one train/test split, we split the data into K=5 parts ("folds").
# Each fold takes one turn as the test set while the other 4 train the model.
# The average of the 5 scores is a more reliable estimate than a single split.
# Run with:  python kfold_cv.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

X, y = load_breast_cancer(return_X_y=True)


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(30,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Stratified = every fold keeps the same ratio of class 0 and class 1
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
    # Scale inside the loop, fitted on this fold's training part only (no data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    model = build_model()  # a fresh, untrained model for every fold
    model.fit(X_train, y[train_idx], epochs=20, batch_size=16, verbose=0)
    _, acc = model.evaluate(X_test, y[test_idx], verbose=0)
    scores.append(acc)
    print(f"Fold {fold}: accuracy = {acc:.3f}")

print(f"\nMean accuracy: {np.mean(scores):.3f}  (+/- {np.std(scores):.3f})")
