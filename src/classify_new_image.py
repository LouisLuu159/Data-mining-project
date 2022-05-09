import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from PIL import Image
from cnn_main import CNNet
from pathlib import Path

model = CNNet(6)
checkpoint = torch.load(
    Path('D:\\Data_mining\\Kaggle-Flower-Recognition-CNN\\src\\18.model'))
model.load_state_dict(checkpoint)
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model.eval()
image_file_name = "astilbe\\4762758746_72eaac60df_c.jpg"
image = Image.open(Path(
    f'D:\\Data_mining\\Dataset\\val\\{image_file_name}'))
input_loader = DataLoader(dataset=image, batch_size=1,
                          shuffle=False, num_workers=5)
input_loader = trans(image)
input_loader = input_loader.view(1, 3, 32, 32)

output = model(input_loader)

prediction = output.data.numpy().argmax()
print(prediction)

if (prediction == 0):
    print('astilbe')
if (prediction == 1):
    print('bellflower')
if (prediction == 2):
    print('black-eyed susan')
if (prediction == 3):
    print('calendula')
if (prediction == 4):
    print('california poppy')
if (prediction == 5):
    print('tulip')
