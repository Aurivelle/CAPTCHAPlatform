
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm



BATCH_SIZE = 32       
NUM_WORKERS = 2       
EPOCHS = 10           
LEARNING_RATE = 1e-3  
SEED = 42

torch.manual_seed(SEED)



transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG 輸入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
testloader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"CIFAR-100 loaded. #train: {len(trainset)}, #test: {len(testset)}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.vgg16(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False  # Freeze convolutional layers

model.classifier[6] = nn.Linear(4096, 100)  # Output for 100 classes
model = model.to(device)


optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_acc_list, test_acc_list, train_loss_list, test_loss_list = [], [], [], []
best_acc = 0
best_model_wts = None



for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_acc_list.append(epoch_acc)
    train_loss_list.append(epoch_loss)

    
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(testloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Test ]"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            running_loss += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    test_acc_list.append(epoch_acc)
    test_loss_list.append(epoch_loss)

    print(f"Epoch {epoch+1}: Train acc={train_acc_list[-1]:.4f}, Test acc={test_acc_list[-1]:.4f}")

    # Save best model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()


if best_model_wts is not None:
    torch.save(best_model_wts, "vgg16_cifar100_best.pt")
    print(f"Best model saved with acc={best_acc:.4f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_acc_list, label='Train Acc')
plt.plot(test_acc_list,  label='Test Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.subplot(1,2,2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list,  label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model.load_state_dict(torch.load("vgg16_cifar100_best.pt"))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in testloader:
        x = x.to(device)
        out = model(x)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10,10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trainset.classes)
disp.plot(ax=ax, values_format="d", cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix (Test set)")
plt.show()

