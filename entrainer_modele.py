import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

DOSSIER_DATASET = r"C:\Users\pc\Desktop\RCP_simulator\nn\dataset"
MODELE_SORTIE   = r"C:\Users\pc\Desktop\RCP_simulator\nn\modele_rotation.pth"

# Transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Charger dataset
train_data = datasets.ImageFolder(os.path.join(DOSSIER_DATASET, "train"), transform=transform_train)
val_data   = datasets.ImageFolder(os.path.join(DOSSIER_DATASET, "val"),   transform=transform_val)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False)

print(f"Train : {len(train_data)} images")
print(f"Val   : {len(val_data)} images")
print(f"Classes : {train_data.classes}")

# Modèle ResNet18 pré-entraîné
modele = models.resnet18(pretrained=True)
modele.fc = nn.Linear(modele.fc.in_features, 4)  # 4 classes : 0, 90, 180, 270

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
modele = modele.to(device)

# Entraînement
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modele.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

EPOCHS = 20

for epoch in range(EPOCHS):
    # Train
    modele.train()
    loss_total = 0
    correct    = 0
    total      = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = modele(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    acc_train = 100 * correct / total

    # Validation
    modele.eval()
    correct_val = 0
    total_val   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = modele(images)
            _, predicted = outputs.max(1)
            correct_val += predicted.eq(labels).sum().item()
            total_val   += labels.size(0)

    acc_val = 100 * correct_val / total_val
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_total/len(train_loader):.3f} | Train: {acc_train:.1f}% | Val: {acc_val:.1f}%")

# Sauvegarder
torch.save(modele.state_dict(), MODELE_SORTIE)
print(f"\nModele sauvegarde : {MODELE_SORTIE}")
print("Entrainement termine !")