# Train the SmallCNN on CIFAR-10 (60,000 small colour photos in 10 classes).
# The dataset (~170 MB) downloads into ./data the first time.
# Run with:  python train.py

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallCNN

EPOCHS = 3
HERE = Path(__file__).parent

# ToTensor: image -> tensor with values 0..1.  Normalize: shift values to roughly -1..1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_data = datasets.CIFAR10(HERE / "data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(HERE / "data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

model = SmallCNN()
loss_fn = nn.CrossEntropyLoss()  # combines softmax + negative log-likelihood
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, EPOCHS + 1):
    model.train()  # turns Dropout on
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()            # 1. clear old gradients
        outputs = model(images)          # 2. forward pass: get predictions
        loss = loss_fn(outputs, labels)  # 3. how wrong are we?
        loss.backward()                  # 4. backpropagation: compute gradients
        optimizer.step()                 # 5. update the weights
        total_loss += loss.item()
    print(f"Epoch {epoch}: average loss = {total_loss / len(train_loader):.3f}")

# Test accuracy
model.eval()  # turns Dropout off
correct = 0
with torch.no_grad():  # no gradients needed for testing -> faster
    for images, labels in test_loader:
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()
print(f"Test accuracy: {correct / len(test_data):.3f}")

torch.save(model.state_dict(), HERE / "cifar_cnn.pth")
print("Saved model to cifar_cnn.pth")
