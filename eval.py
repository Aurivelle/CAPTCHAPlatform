import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(4096, 100)
model.load_state_dict(torch.load("vgg16_cifar100_best.pt", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


from torchvision.datasets import CIFAR100

label_names = CIFAR100(root="./data", train=False, download=True).classes


def predict_fn(img: Image.Image) -> str:
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    pred = out.argmax(1).item()
    return label_names[pred]


from metrics import evaluate_folder


img_dir = "dataset/"
label_path = "dataset/labels.txt"


result = evaluate_folder(predict_fn, img_dir, label_path)
print("✅ 評估結果：", result)
