from ultralytics import YOLO
import os
import shutil

DOSSIER_PHOTOS  = r"C:\Users\pc\Desktop\RCP_simulator\nn\photo"
DOSSIER_LABELS  = r"C:\Users\pc\Desktop\RCP_simulator\nn\labels_portes"
DOSSIER_DATASET = r"C:\Users\pc\Desktop\RCP_simulator\nn\dataset_portes"

# Créer structure dataset
os.makedirs(f"{DOSSIER_DATASET}/train/images", exist_ok=True)
os.makedirs(f"{DOSSIER_DATASET}/train/labels", exist_ok=True)
os.makedirs(f"{DOSSIER_DATASET}/val/images",   exist_ok=True)
os.makedirs(f"{DOSSIER_DATASET}/val/labels",   exist_ok=True)

# Copier photos annotées
photos_annotees = []
for f in os.listdir(DOSSIER_LABELS):
    if f.endswith(".txt"):
        nom_base = os.path.splitext(f)[0]
        for ext in [".jpg", ".jpeg", ".png"]:
            chemin_photo = os.path.join(DOSSIER_PHOTOS, nom_base + ext)
            if os.path.exists(chemin_photo):
                photos_annotees.append((chemin_photo,
                                        os.path.join(DOSSIER_LABELS, f)))
                break

print(f"{len(photos_annotees)} photos annotees trouvees")

if len(photos_annotees) == 0:
    print("ERREUR — Aucune photo trouvee !")
    print(f"Verifie que les photos sont dans : {DOSSIER_PHOTOS}")
    print(f"Verifie que les labels sont dans : {DOSSIER_LABELS}")
    exit()

# 80% train / 20% val
split = max(1, int(len(photos_annotees) * 0.8))
train = photos_annotees[:split]
val   = photos_annotees[split:] if len(photos_annotees[split:]) > 0 else photos_annotees[:1]

for chemin_photo, chemin_label in train:
    shutil.copy(chemin_photo, f"{DOSSIER_DATASET}/train/images/")
    shutil.copy(chemin_label, f"{DOSSIER_DATASET}/train/labels/")

for chemin_photo, chemin_label in val:
    shutil.copy(chemin_photo, f"{DOSSIER_DATASET}/val/images/")
    shutil.copy(chemin_label, f"{DOSSIER_DATASET}/val/labels/")

print(f"Train : {len(train)} photos")
print(f"Val   : {len(val)} photos")

# Créer yaml
yaml_content = f"""path: {DOSSIER_DATASET.replace(chr(92), '/')}
train: train/images
val: val/images

nc: 1
names: ['door']
"""

with open("portes.yaml", "w") as f:
    f.write(yaml_content)

print("Dataset cree — lancement entrainement...")

# Entraîner
modele = YOLO("yolov8n.pt")
modele.train(
    data="portes.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="modele_portes",
    patience=10,
)

print("Entrainement termine !")
print("Modele sauvegarde : runs/detect/modele_portes/weights/best.pt")