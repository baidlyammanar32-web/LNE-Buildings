from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageOps
import os

MODELE_PATH = r"C:\Users\pc\Desktop\RCP_simulator\nn\runs\detect\modele_portes\weights\best.pt"
modele_portes = YOLO(MODELE_PATH)

def detecter_portes(chemin_photo):
    image = Image.open(chemin_photo).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image_np  = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    resultats = modele_portes(image_bgr, verbose=False, conf=0.3)

    portes = []
    for r in resultats:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            h    = y2 - y1

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image_bgr, f"Porte | {h}px | {conf:.0%}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
            cx = (x1 + x2) // 2
            cv2.arrowedLine(image_bgr, (cx, y2), (cx, y1),
                           (255, 0, 255), 1, tipLength=0.05)
            portes.append(h)
            print(f"Porte : {h}px | confiance : {conf:.0%}")

    image_result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(image_result).save("resultat_portes.jpg")
    print(f"{len(portes)} porte(s) detectee(s)")
    print("Image sauvegardee : resultat_portes.jpg")
    return portes

# Test
PHOTO_TEST = r"C:\Users\pc\Desktop\RCP_simulator\nn\photo\ACNGL_2026-05-11_11-07-19.jpg"
if os.path.exists(PHOTO_TEST):
    detecter_portes(PHOTO_TEST)
else:
    # Prendre la première photo du dossier
    dossier = r"C:\Users\pc\Desktop\RCP_simulator\nn\photo"
    photos  = [f for f in os.listdir(dossier) if f.endswith(".jpg")]
    if photos:
        detecter_portes(os.path.join(dossier, photos[0]))