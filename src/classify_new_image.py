from logging import root
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from cnn_main import CNNet
from pathlib import Path
import os

model = CNNet(6)
checkpoint = torch.load(
    Path('D:\\Data_mining\\Code\\Data-mining-project\\src\\14.model'))
model.load_state_dict(checkpoint)

trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model.eval()


def get_predicted_label(path):
    image = Image.open(Path(path))

    input_loader = DataLoader(dataset=image, batch_size=1,
                              shuffle=False, num_workers=2)
    input_loader = trans(image)
    input_loader = input_loader.view(1, 3, 32, 32)

    output = model(input_loader)

    prediction = output.data.numpy().argmax()

    labels = ['astilbe', 'bellflower', 'black-eyed susan',
              'calendula', 'california poppy', 'tulip']

    return labels[prediction]


predicted_labels = []


rootdir = 'D:\\Data_mining\\val'  # test folder
for rootdir, dirs, files in os.walk(rootdir):
    for filename in files:
        predicted_label = get_predicted_label(os.path.join(rootdir, filename))
        predicted_labels.append(predicted_label)

print(f"Predict label: {predicted_labels}")
