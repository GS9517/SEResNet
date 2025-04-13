import sys
if len(sys.argv) > 2:
    image_path = sys.argv[1]
    weights_path = sys.argv[2]
else:
    print("Usage: python detect.py <image_path> <weights_path>")
    print("Exiting.")
    sys.exit(1)

import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(weights_path, weights_only=False, map_location=device)

generic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
               std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):

    image = Image.open(image_path).convert('RGB')
    input_tensor = generic_transform(image)
    
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        curr_time = time.time()
        output = model(input_batch)
        elapsed_time = time.time() - curr_time
        print(f"Prediction time: {elapsed_time:.4f} seconds")
    
    probabilities = torch.nn.functional.softmax(output, dim=1)
    return probabilities.cpu().numpy().squeeze()


class_probs = predict_image(image_path)

classes = "Agriculture  Airport  Beach  City  Desert  Forest  Grassland  Highway  Lake  Mountain  Parking  Port  Railway  Residential  River".split() 

sorted_indices = np.argsort(-class_probs)

print("Possibilities:")
i = 1
for idx in sorted_indices:
    print(f"{i} - {classes[idx].ljust(15)}: {class_probs[idx]*100:.2f}%")
    i += 1
    
