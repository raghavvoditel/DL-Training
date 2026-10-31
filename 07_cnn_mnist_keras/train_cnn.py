# CNN (Convolutional Neural Network) that reads handwritten digits 0-9 (MNIST).
# Saves the trained model to digit_cnn.keras so predict_app.py can use it.
# Run with:  python train_cnn.py

from pathlib import Path

from tensorflow import keras

# 1. Load MNIST: 60,000 training and 10,000 test images, each 28x28 pixels, grayscale
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Scale pixels from 0..255 to 0..1 and add a "channel" dimension: (28, 28) -> (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# 3. Build the CNN
#    Conv2D  = slides small 3x3 filters over the image to detect edges, curves, ...
#    MaxPool = shrinks the image by keeping the strongest value in each 2x2 block
#    Flatten = turns the 2D feature maps into one long list for the Dense layers
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(16, (3, 3), activation="relu"),   # -> 26x26x16
    keras.layers.MaxPooling2D((2, 2)),                     # -> 13x13x16
    keras.layers.Conv2D(32, (3, 3), activation="relu"),   # -> 11x11x32
    keras.layers.MaxPooling2D((2, 2)),                     # -> 5x5x32
    keras.layers.Flatten(),                                # -> 800
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),                             # randomly switch off 20% of neurons while training
    keras.layers.Dense(10, activation="softmax"),          # 10 probabilities, one per digit
])

# sparse_categorical_crossentropy: labels are plain numbers 0..9 (not one-hot)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.4f}")

save_path = Path(__file__).parent / "digit_cnn.keras"
model.save(save_path)
print(f"Model saved to {save_path}")
