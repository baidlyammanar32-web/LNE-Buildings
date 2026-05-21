import os
from PIL import Image

# Dossier de tes photos originales
DOSSIER_PHOTOS = r"C:\Users\pc\Desktop\RCP_simulator\nn"

# Dossier dataset à créer
DOSSIER_DATASET = r"C:\Users\pc\Desktop\RCP_simulator\nn\dataset"

angles = {
    "0":   0,
    "90":  90,
    "180": 180,
    "270": 270,
}

# Créer les dossiers
for label in angles:
    os.makedirs(os.path.join(DOSSIER_DATASET, "train", label), exist_ok=True)
    os.makedirs(os.path.join(DOSSIER_DATASET, "val",   label), exist_ok=True)

# Lister toutes les photos
photos = [f for f in os.listdir(DOSSIER_PHOTOS) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]

print(f"{len(photos)} photos trouvees")

for i, nom_photo in enumerate(photos):
    chemin = os.path.join(DOSSIER_PHOTOS, nom_photo)
    image  = Image.open(chemin).convert("RGB")

    for label, angle in angles.items():
        image_rot = image.rotate(angle, expand=True)
        image_rot = image_rot.resize((224, 224))

        # 80% train / 20% val
        dossier = "train" if i % 5 != 0 else "val"
        nom_sortie = f"{nom_photo.replace('.jpg','').replace('.jpeg','').replace('.png','')}_{label}.jpg"
        chemin_sortie = os.path.join(DOSSIER_DATASET, dossier, label, nom_sortie)
        image_rot.save(chemin_sortie)

print("Dataset cree avec succes !")
print(f"Dossier : {DOSSIER_DATASET}")