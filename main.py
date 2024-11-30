import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Charger le modèle de classification des marques
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_classes = 19
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("trained_model.pth"))
model = model.to(device)
model.eval()

# Liste des classes (marques de voitures)
class_names = [
    "Audi", "Bentley", "Benz", "Bmw", "Cadillac", "Dodge", "Ferrari", "Ford", "Ford_mustang", "Kia",
    "Lamborghini", "Lexus", "Maserati", "Porsche", "Rolls_royce", "Tesla", "Toyota", "alfa_romeo", "hyundai"
]

# Transformations d'image
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Détecter les voitures avec YOLOv5
def detect_vehicle(image_path):
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    results = model_yolo(image_path)
    detections = results.xyxy[0].cpu().numpy()  # Coordonnées des boîtes détectées
    return detections

# Prédire la marque de la voiture
def predict_image(image):
    image = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Fonction principale
def predict_image_with_box(image_path):
    # Charger l'image
    original_image = cv2.imread(image_path)
    detections = detect_vehicle(image_path)

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if cls == 2 or cls == 7:  # ID pour "car" ou "truck" dans YOLO
            cropped_img = original_image[int(y1):int(y2), int(x1):int(x2)]
            pil_image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            class_name = predict_image(pil_image)

            # Dessiner l'encadré et le texte
            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(original_image, class_name, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Redimensionner l'image pour l'affichage si elle est trop grande ou trop petite
    display_width = 800  # Largeur désirée pour l'affichage
    scale = display_width / original_image.shape[1]
    resized_image = cv2.resize(original_image, (display_width, int(original_image.shape[0] * scale)))

    # Afficher l'image redimensionnée
    cv2.imshow("Image with Prediction", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exemple d'appel
image_path = "C:/Users/Mathiss/PycharmProjects/ModelCar_test/test/image5.jpeg"
predict_image_with_box(image_path)

"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Configurations
data_dir = "C:/Users/Mathiss/PycharmProjects/ModelCar_test/organized_dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
batch_size = 32
num_epochs = 10
num_classes = 19  # Nombre de marques
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Datasets and Dataloaders
image_datasets = {
    "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
    "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
}
dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
    "test": DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False)
}

# Load Pretrained Model and Modify
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer
model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model():
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Training Loss: {running_loss / len(dataloaders['train'])}")

# Evaluation
def evaluate_model():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=image_datasets["test"].classes))

# Save the model
def save_model():
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé sous {model_path}")

# Run Training and Evaluation
train_model()
evaluate_model()

# Save the model after training
save_model()
"""
