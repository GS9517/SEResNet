import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.utils.data.dataset
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from SEResNet10 import SEResNet10
from TransformedSubset import TransformedSubset

path = "./Aerial_Landscapes"
count = 1
while os.path.exists(f"runs/test{count}"):
    count += 1
os.makedirs(f"runs/test{count}", exist_ok=False)

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
test_dataset = TransformedSubset(test_dataset, generic_transform)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SEResNet10(num_classes=15).to(device)
model.load_state_dict(torch.load('runs/train3/models/best_model.pth'))
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
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
test_accuracy = 100 * correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.2f}%")
        

cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix for Test Set\nAccuracy: {test_accuracy:.2f}%')
plt.savefig(f'runs/test{count}/confusion_matrix.png')
plt.close()