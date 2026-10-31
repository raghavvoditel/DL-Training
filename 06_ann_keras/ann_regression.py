# ANN for REGRESSION (predicting a number instead of a class).
# Task: predict how much a patient's diabetes progresses in one year from 10 health values.
# The dataset is built into scikit-learn.
# Run with:  python ann_regression.py

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

X, y = load_diabetes(return_X_y=True)  # 442 rows x 10 features, y is a number (25..346)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.Input(shape=(10,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1),  # no activation: the output can be any number
])

# mse = mean squared error, the usual loss for regression
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

# EarlyStopping stops training when validation loss stops improving (avoids overfitting)
stop_early = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.1,
          callbacks=[stop_early], verbose=0)

pred = model.predict(X_test, verbose=0).flatten()
print(f"Mean absolute error: {mean_absolute_error(y_test, pred):.1f}")
print(f"R2 score: {r2_score(y_test, pred):.3f}   (1.0 is perfect, 0 means no better than the average)")

for p, real in zip(pred[:5], y_test[:5]):
    print(f"predicted {p:6.1f}   real {real:6.1f}")
