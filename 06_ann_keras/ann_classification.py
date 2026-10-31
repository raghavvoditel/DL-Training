# ANN (Artificial Neural Network) for CLASSIFICATION.
# Task: predict if a breast tumour is malignant (0) or benign (1) from 30 measurements.
# The dataset comes built into scikit-learn, so nothing to download.
# Run with:  python ann_classification.py

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# 1. Load data: X = features (569 rows x 30 columns), y = labels (0 or 1)
X, y = load_breast_cancer(return_X_y=True)

# 2. Split: 80% for training, 20% kept aside for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 3. Scale: give every feature mean 0 and std 1, so big numbers don't dominate.
#    fit only on training data, then apply the same scaling to the test data.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the network: 30 inputs -> 16 neurons -> 8 neurons -> 1 output
model = keras.Sequential([
    keras.Input(shape=(30,)),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),  # sigmoid gives a probability 0..1
])

# binary_crossentropy is the loss for yes/no problems
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 5. Train
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=2)

# 6. Test on data the model has never seen
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy:.3f}")

# 7. Predict the first 5 test samples
probs = model.predict(X_test[:5], verbose=0).flatten()
for p, real in zip(probs, y_test[:5]):
    print(f"probability benign = {p:.2f} -> predicted {int(p > 0.5)}, real {real}")
