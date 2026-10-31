# The CNN model, written in PyTorch. Used by train.py and predict.py.

import torch.nn as nn

CLASSES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: a colour image, 3 channels (red, green, blue), 32x32 pixels
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # -> 32 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 32 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 64 x 8 x 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # -> 64*8*8 = 4096 numbers
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASSES)),  # one score per class (softmax is inside the loss)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
