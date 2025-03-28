# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CURz-KkYhkhD7PqrXqdAroc3hteQ_2_-
"""
import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.utils.data.dataset
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from SEResNet10 import SEResNet10
from TransformedSubset import TransformedSubset

path = "./Aerial_Landscapes"
count = 1
while os.path.exists(f"runs/train{count}"):
    count += 1
os.makedirs(f"runs/train{count}", exist_ok=False)


class RandomMask(nn.Module):
    def __init__(self, num_blocks=3, block_size=50, p=0.5):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            # 获取Tensor格式的尺寸
            c, h, w = img.shape
            
            for _ in range(self.num_blocks):
                # random block position
                x = random.randint(0, h - self.block_size)
                y = random.randint(0, w - self.block_size)
                
                # random mask type
                mask_type = random.choice([0, 1, 2])
                
                if mask_type == 0:  # gray block
                    img[:, x:x+self.block_size, y:y+self.block_size] = 0.5 # 0.5是灰色
                elif mask_type == 1:  # random noise
                    noise = torch.rand_like(img[:, x:x+self.block_size, y:y+self.block_size])
                    img[:, x:x+self.block_size, y:y+self.block_size] = noise
                else: # 不操作
                    pass
        return img

# Define the transformations for the training sets
train_transform = transforms.Compose([
    # PIL
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    # transforms.RandomPerspective(distortion_scale=0.5, p=1.0), 
    # 别开这个选项 模型完全没有空间感知能力 开了包炸的
    transforms.RandomRotation(15),
    
    transforms.ToTensor(),
    # Augmentations for Tensor
    RandomMask(num_blocks=2, block_size=32, p=0.5),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

generic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=path, transform=None)
dataset_size = len(dataset)

train_size = int(0.7 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Apply the transforms to the datasets
train_dataset = TransformedSubset(train_dataset, train_transform)
val_dataset = TransformedSubset(val_dataset, generic_transform)
test_dataset = TransformedSubset(test_dataset, generic_transform)

def save_samples(dataset, title, filename_prefix, num_samples=5):
    os.makedirs(f"runs/train{count}/samples", exist_ok=True)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # Denormalize
        img = torch.clamp(img, 0.0, 1.0)  # Clamp values to [0, 1]
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Class: {label}")
    plt.savefig(f"runs/train{count}/samples/{filename_prefix}.png")
    plt.close()

# Save samples from each dataset
save_samples(train_dataset, "Training Set Samples", "train_samples")
save_samples(val_dataset, "Validation Set Samples", "val_samples")
save_samples(test_dataset, "Test Set Samples", "test_samples")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SEResNet10(num_classes=15).to(device)
criterion = nn.CrossEntropyLoss()
num_epochs = 150
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True) # Yolo v11's parameter, I do not know why, dont modify it, it is perfect
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.3*num_epochs), int(0.6* num_epochs), int(0.9*num_epochs)], gamma=0.1)
# Print the total number of parameters in the modeld
total_params = sum(p.numel() for p in model.parameters())

print(f"Total number of parameters: {total_params}")


os.makedirs(f"runs/train{count}/models", exist_ok=True)
# TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/train{count}/logs")

csv_file = f"runs/train{count}/training_log.csv"
with open(csv_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Epoch", "Training Loss", "Validation Accuracy"])

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    best_accuracy = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / train_size
    

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probs = F.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    # Calculate validation loss
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_loss /= val_size

    writer.add_scalar("Accuracy/Validation", accuracy, epoch)
    writer.add_scalar("Loss/Training", epoch_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)

    # Log metrics to CSV
    with open(csv_file, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([epoch + 1, epoch_loss, accuracy])

    print(f"Validation: Accuracy: {accuracy:.2f}%, Loss: {val_loss:.2f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), f'runs/train{count}/models/best_model.pth')

    # Save model weights every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'runs/train{count}/models/model_epoch_{epoch+1}_{accuracy:.2f}.pth')

writer.close()

model.eval()
correct_test = 0
total_test = 0
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix for Validation Set (Epoch {epoch+1})')
plt.savefig(f'runs/train{count}/confusion_matrix_val_epoch_{epoch+1}.png')
plt.close()