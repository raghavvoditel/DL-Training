# Classify your own photo with the trained CNN.
# First run train.py, then:  python predict.py path/to/photo.jpg

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import CLASSES, SmallCNN

# Same preprocessing as training, plus resizing any photo to 32x32
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

model = SmallCNN()
model.load_state_dict(torch.load(Path(__file__).parent / "cifar_cnn.pth"))
model.eval()

image = Image.open(sys.argv[1]).convert("RGB")
batch = transform(image).unsqueeze(0)  # add a batch dimension: (3,32,32) -> (1,3,32,32)

with torch.no_grad():
    probs = torch.softmax(model(batch), dim=1)[0]

top3 = probs.argsort(descending=True)[:3]
for i in top3:
    print(f"{CLASSES[i]:>10}: {probs[i] * 100:.1f}%")
